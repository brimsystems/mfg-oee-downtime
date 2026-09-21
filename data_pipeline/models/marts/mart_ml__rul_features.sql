-- ML feature table for the Remaining Useful Life model.
-- Grain: one row per (machine_id, observation_date, shift) across operating days.
--
-- Target (target_days_to_failure): days from the observation date to the next
-- unplanned maintenance event in the CMMS, capped at the RUL horizon. Where no
-- future failure exists within the data window the observation is right-censored
-- (is_censored = true) and the target is set to the horizon cap. Features are
-- all backward-looking as of the observation date so there is no target leakage.
--
-- Condition-monitoring layer: daily sensor readings (vibration, bearing temp,
-- spindle power, hydraulic pressure) are rolled up into 7-day means and anomaly
-- z-scores (7-day mean vs the machine's trailing 30-day baseline). These lead
-- failures because sensor readings drift ahead of an unplanned event.

with machines as (

    select machine_id, machine_type, controller_type, machine_age_years
    from {{ ref('int_dim_machines') }}

),

daily_states as (

    select
        machine_id,
        event_date,
        sum(state_duration_minutes) filter (where machine_state = 'UNPLANNED_DOWN') as unplanned_down_minutes,
        count(*)                    filter (where machine_state = 'ALARM')          as alarm_count,
        sum(state_duration_minutes) filter (where machine_state = 'RUNNING')        as run_minutes,
        sum(state_duration_minutes)                                                 as scheduled_minutes
    from {{ ref('int_fct_machine_states') }}
    group by 1, 2

),

failures as (

    select machine_id, event_date as failure_date, failure_code
    from {{ ref('int_fct_maintenance_events') }}
    where is_unplanned_repair

),

pms as (

    select machine_id, pm_completed_date, days_overdue
    from {{ ref('int_fct_maintenance_events') }}
    where is_planned_pm and pm_completed_date is not null

),

sensors as (

    select machine_id, reading_date, vibration_rms_mm_s, bearing_temp_c,
           spindle_power_kw, hydraulic_pressure_bar
    from {{ ref('stg_sensors__readings') }}

),

observation_dates as (

    select distinct calendar_date as observation_date, shift_code as shift
    from {{ ref('int_dim_shifts') }}

),

rolling as (

    select
        md.machine_id,
        md.observation_date,
        coalesce(sum(d.unplanned_down_minutes)
                 filter (where d.event_date > md.observation_date - 7), 0) / 60.0   as rolling_7d_unplanned_downtime_hours,
        coalesce(sum(d.alarm_count)
                 filter (where d.event_date > md.observation_date - 7), 0)          as rolling_7d_alarm_count,
        coalesce(sum(d.alarm_count)
                 filter (where d.event_date > md.observation_date - 30), 0)         as rolling_30d_alarm_count,
        sum(d.run_minutes) filter (where d.event_date > md.observation_date - 30)
            / nullif(sum(d.scheduled_minutes)
                     filter (where d.event_date > md.observation_date - 30), 0)     as rolling_30d_utilization_rate
    from (
        select m.machine_id, o.observation_date
        from machines m
        cross join (select distinct observation_date from observation_dates) o
    ) md
    left join daily_states d
        on d.machine_id = md.machine_id
       and d.event_date <= md.observation_date
       and d.event_date >  md.observation_date - 30
    group by 1, 2

),

sensor_rolling as (

    select
        md.machine_id,
        md.observation_date,
        avg(s.vibration_rms_mm_s)     filter (where s.reading_date > md.observation_date - 7) as vib_7d,
        avg(s.vibration_rms_mm_s)                                                             as vib_30d,
        stddev_samp(s.vibration_rms_mm_s)                                                     as vib_30d_std,
        avg(s.bearing_temp_c)         filter (where s.reading_date > md.observation_date - 7) as temp_7d,
        avg(s.bearing_temp_c)                                                                 as temp_30d,
        stddev_samp(s.bearing_temp_c)                                                         as temp_30d_std,
        avg(s.spindle_power_kw)       filter (where s.reading_date > md.observation_date - 7) as power_7d,
        avg(s.spindle_power_kw)                                                               as power_30d,
        stddev_samp(s.spindle_power_kw)                                                       as power_30d_std,
        avg(s.hydraulic_pressure_bar) filter (where s.reading_date > md.observation_date - 7) as press_7d,
        avg(s.hydraulic_pressure_bar)                                                         as press_30d,
        stddev_samp(s.hydraulic_pressure_bar)                                                 as press_30d_std
    from (
        select m.machine_id, o.observation_date
        from machines m
        cross join (select distinct observation_date from observation_dates) o
    ) md
    left join sensors s
        on s.machine_id = md.machine_id
       and s.reading_date <= md.observation_date
       and s.reading_date >  md.observation_date - 30
    group by 1, 2

),

enriched as (

    select
        m.machine_id,
        o.observation_date,
        o.shift,
        m.machine_type,
        m.controller_type,
        m.machine_age_years,

        r.rolling_7d_unplanned_downtime_hours,
        r.rolling_7d_alarm_count,
        r.rolling_30d_alarm_count,
        r.rolling_30d_utilization_rate,

        sr.vib_7d, sr.vib_30d, sr.vib_30d_std,
        sr.temp_7d, sr.temp_30d, sr.temp_30d_std,
        sr.power_7d, sr.power_30d, sr.power_30d_std,
        sr.press_7d, sr.press_30d, sr.press_30d_std,

        date_diff('day',
            (select max(f.failure_date) from failures f
             where f.machine_id = m.machine_id and f.failure_date < o.observation_date),
            o.observation_date)                                             as days_since_last_unplanned_failure,

        (select f.failure_code from failures f
         where f.machine_id = m.machine_id and f.failure_date < o.observation_date
         order by f.failure_date desc limit 1)                             as last_failure_mode,

        date_diff('day',
            (select max(p.pm_completed_date) from pms p
             where p.machine_id = m.machine_id and p.pm_completed_date <= o.observation_date),
            o.observation_date)                                             as days_since_last_pm,

        (select count(*) from pms p
         where p.machine_id = m.machine_id
           and p.pm_completed_date <= o.observation_date
           and p.pm_completed_date >  o.observation_date - 182
           and p.days_overdue > 0)                                         as count_late_pms_last_6m,

        (select min(f.failure_date) from failures f
         where f.machine_id = m.machine_id and f.failure_date > o.observation_date)  as next_failure_date

    from machines m
    cross join observation_dates o
    left join rolling r
        on r.machine_id = m.machine_id and r.observation_date = o.observation_date
    left join sensor_rolling sr
        on sr.machine_id = m.machine_id and sr.observation_date = o.observation_date

),

final as (

    select
        {{ dbt_utils.generate_surrogate_key(['machine_id', 'observation_date', 'shift']) }}  as rul_key,
        machine_id,
        observation_date,
        shift,
        machine_type,
        controller_type,
        machine_age_years,

        rolling_7d_unplanned_downtime_hours,
        rolling_7d_alarm_count,
        rolling_30d_alarm_count,
        rolling_30d_utilization_rate,
        days_since_last_unplanned_failure,
        last_failure_mode,
        days_since_last_pm,
        case
            when days_since_last_pm is null then null
            else greatest(0, days_since_last_pm - {{ var('pm_interval_days') }})
        end                                                                 as days_overdue_for_pm,
        count_late_pms_last_6m,

        -- Condition-monitoring features (backward-looking).
        round(vib_7d, 3)                                                    as vibration_7d_mean,
        round(temp_7d, 2)                                                   as bearing_temp_7d_mean,
        round(power_7d, 3)                                                  as spindle_power_7d_mean,
        round(press_7d, 2)                                                  as hydraulic_pressure_7d_mean,
        round((vib_7d - vib_30d)   / nullif(vib_30d_std, 0), 3)             as vibration_anomaly,
        round((temp_7d - temp_30d) / nullif(temp_30d_std, 0), 3)           as bearing_temp_anomaly,
        round((power_7d - power_30d) / nullif(power_30d_std, 0), 3)        as spindle_power_anomaly,
        round((press_7d - press_30d) / nullif(press_30d_std, 0), 3)       as hydraulic_pressure_anomaly,
        round(greatest(
            coalesce((vib_7d - vib_30d)     / nullif(vib_30d_std, 0), 0),
            coalesce((temp_7d - temp_30d)   / nullif(temp_30d_std, 0), 0),
            coalesce((power_7d - power_30d) / nullif(power_30d_std, 0), 0),
            coalesce(abs((press_7d - press_30d) / nullif(press_30d_std, 0)), 0)
        ), 3)                                                               as sensor_anomaly_score,

        (next_failure_date is null)                                         as is_censored,
        least(
            {{ var('rul_horizon_days') }},
            coalesce(date_diff('day', observation_date, next_failure_date),
                     {{ var('rul_horizon_days') }})
        )                                                                   as target_days_to_failure

    from enriched

)

select * from final
