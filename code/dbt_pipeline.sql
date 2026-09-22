-- dbt transformation layer (DuckDB): staging, intermediate and marts. Sources are read from generated CSV extracts.

-- ########################################################################
-- STAGING LAYER
-- ########################################################################

-- ====================================================================
-- model: stg_cmms__maintenance_records
-- ====================================================================

with source as (

    select * from {{ source('cmms', 'maintenance_records') }}

),

staged as (

    select
        maintenance_id,
        machine_id,
        maintenance_type,

        -- Populated for unplanned repairs only; null for PM and inspections.
        failure_code,

        cast(work_order_open_date as timestamp)     as work_order_open_date,
        cast(work_order_close_date as timestamp)    as work_order_close_date,
        cast(downtime_hours as double)              as downtime_hours,
        cast(technician_empid as integer)           as technician_empid,
        parts_consumed,
        resolution_notes,

        -- Ad-hoc PMs are logged with a completion but no scheduled date, so the
        -- scheduled date and days_overdue are null for those rows. days_overdue
        -- is retained from source as (completed - scheduled); negative is early.
        cast(pm_scheduled_date as date)             as pm_scheduled_date,
        cast(pm_completed_date as date)             as pm_completed_date,
        cast(days_overdue as integer)               as days_overdue

    from source

)

select * from staged

-- ====================================================================
-- model: stg_erp__work_orders
-- ====================================================================

with source as (

    select * from {{ source('erp', 'work_orders') }}

),

staged as (

    select
        work_order_id,
        machine_id,

        -- Payroll number, matching the MachineMetrics identifier scheme and
        -- reconciled to the HR operator_id in the intermediate layer.
        cast(operator_empid as integer)     as operator_empid,

        part_number,
        customer_id,
        cast(scheduled_start as timestamp)  as scheduled_start,
        cast(scheduled_end as timestamp)    as scheduled_end,
        cast(actual_start as timestamp)     as actual_start,

        -- Null until the job completes; open jobs carry no actual end or hours.
        cast(actual_end as timestamp)       as actual_end,
        cast(scheduled_hours as double)     as scheduled_hours,
        cast(actual_hours as double)        as actual_hours,

        cast(setup_hours_actual as double)  as setup_hours_actual,
        job_status,
        material_type

    from source

)

select * from staged

-- ====================================================================
-- model: stg_hr__operators
-- ====================================================================

with source as (

    select * from {{ source('hr', 'operators') }}

),

staged as (

    select
        operator_id,
        operator_name,

        -- Payroll number carried by ADP. This is the only operator key present
        -- in the ERP and MachineMetrics extracts, so it is the bridge used to
        -- reconcile those systems back to operator_id in the intermediate layer.
        cast(employee_number as integer)           as employee_number,

        shift,
        role,
        cast(hire_date as date)                    as hire_date,
        certification,
        cast(certification_expiry_date as date)    as certification_expiry_date

    from source

)

select * from staged

-- ====================================================================
-- model: stg_machinemetrics__machines
-- ====================================================================

with source as (

    select * from {{ source('machinemetrics', 'machines') }}

),

staged as (

    select
        machine_id,
        machine_type,
        controller_type,
        location_cell,
        cast(machine_age_years as integer)  as machine_age_years,
        cast(max_spindle_rpm as integer)    as max_spindle_rpm,
        cast(install_date as date)          as install_date

    from source

)

select * from staged

-- ====================================================================
-- model: stg_machinemetrics__production_events
-- ====================================================================

with source as (

    select * from {{ source('machinemetrics', 'production_events') }}

),

staged as (

    select
        event_id,
        machine_id,
        cast(event_timestamp as timestamp)          as event_timestamp,
        shift,
        machine_state,
        cast(state_duration_minutes as integer)     as state_duration_minutes,

        -- Machine log records the payroll number (ERP identifier scheme), not
        -- the HR operator_id. Renamed here to make the identifier scheme explicit.
        cast(operator_id as integer)                as operator_empid,

        -- Populated only while RUNNING; null in every other state.
        cast(spindle_utilization_pct as double)     as spindle_utilization_pct,

        -- Populated only in the ALARM state.
        alarm_code

    from source

)

select * from staged

-- ====================================================================
-- model: stg_sensors__readings
-- ====================================================================

-- Daily condition-monitoring readings from the retrofit IIoT gateway.
-- Grain: one row per (machine_id, reading_date).

with source as (

    select * from {{ source('sensors', 'sensor_readings') }}

),

staged as (

    select
        machine_id,
        cast(reading_date as date)              as reading_date,
        cast(vibration_rms_mm_s as double)      as vibration_rms_mm_s,
        cast(bearing_temp_c as double)          as bearing_temp_c,
        cast(spindle_power_kw as double)        as spindle_power_kw,
        cast(hydraulic_pressure_bar as double)  as hydraulic_pressure_bar

    from source

)

select * from staged

-- ########################################################################
-- INTERMEDIATE LAYER
-- ########################################################################

-- ====================================================================
-- model: int_dim_machines
-- ====================================================================

-- Conformed machine dimension. Grain: one row per machine_id.
-- machine_id is consistent across the MachineMetrics, ERP, and CMMS extracts,
-- so this is a clean reference with a few derived analytic attributes.

with source as (

    select * from {{ ref('stg_machinemetrics__machines') }}

),

final as (

    select
        machine_id,
        machine_type,
        controller_type,
        location_cell,
        machine_age_years,
        max_spindle_rpm,
        install_date,

        case
            when machine_age_years < 3  then '<3 yrs'
            when machine_age_years <= 7 then '3-7 yrs'
            when machine_age_years <= 10 then '7-10 yrs'
            else '11+ yrs'
        end                                 as age_band,

        (machine_age_years > 9)             as is_aging_asset

    from source

)

select * from final

-- ====================================================================
-- model: int_dim_operators
-- ====================================================================

-- Conformed operator dimension. Grain: one row per operator.
-- Carries both identifier schemes (operator_id and employee_number), making
-- this the crosswalk that reconciles ERP/MachineMetrics payroll numbers back
-- to the HR operator record. Certification currency is evaluated as of the
-- configured as_of_date.

with source as (

    select * from {{ ref('stg_hr__operators') }}

),

final as (

    select
        operator_id,
        operator_name,
        employee_number,
        shift,
        role,
        hire_date,
        certification,
        certification_expiry_date,

        case
            when certification_expiry_date < date '{{ var("as_of_date") }}'
                then 'Lapsed'
            when certification_expiry_date
                 <= date '{{ var("as_of_date") }}' + interval '30 days'
                then 'Expiring Soon'
            else 'Current'
        end                                 as cert_status

    from source

)

select * from final

-- ====================================================================
-- model: int_dim_shifts
-- ====================================================================

-- Shift calendar. Grain: one row per (calendar_date, shift_code) across the
-- operating window (Monday-Saturday, two shifts). Provides explicit shift
-- windows so facts can attribute a timestamp to a shift. Attribution is by
-- start time: work that overruns the shift end (e.g. a job starting 14:00 that
-- runs past 22:00) remains attributed to the shift it started in.

with calendar as (

    select cast(unnest(generate_series(
        timestamp '{{ var("start_date") }}',
        timestamp '{{ var("end_date") }}',
        interval '1 day'
    )) as date)                                     as calendar_date

),

shifts as (

    select 'A' as shift_code, 'Shift A' as shift_label, 6  as start_hour, 14 as end_hour
    union all
    select 'B' as shift_code, 'Shift B' as shift_label, 14 as start_hour, 22 as end_hour

),

final as (

    select
        cast(c.calendar_date as varchar) || '-' || s.shift_code   as shift_key,
        c.calendar_date,
        s.shift_code,
        s.shift_label,
        c.calendar_date + s.start_hour * interval '1 hour'        as shift_start_ts,
        c.calendar_date + s.end_hour   * interval '1 hour'        as shift_end_ts,
        isodow(c.calendar_date)                                   as day_of_week,
        (isodow(c.calendar_date) = 6)                             as is_saturday,
        date_trunc('month', c.calendar_date)                      as calendar_month

    from calendar c
    cross join shifts s
    where isodow(c.calendar_date) <= 6   -- Monday-Saturday

)

select * from final

-- ====================================================================
-- model: int_fct_machine_states
-- ====================================================================

-- Machine state fact enriched with machine and operator dimensions.
-- Grain: one row per event_id. Dimensions are LEFT joined so no state rows are
-- dropped on a dimension miss; join resolution is asserted by not_null tests.
-- The operator join reconciles the payroll number (operator_empid) back to the
-- HR operator record.

with events as (

    select * from {{ ref('stg_machinemetrics__production_events') }}

),

machines as (

    select
        machine_id,
        machine_type,
        controller_type,
        location_cell,
        machine_age_years,
        age_band,
        is_aging_asset
    from {{ ref('int_dim_machines') }}

),

operators as (

    select
        employee_number,
        operator_id,
        operator_name,
        role        as operator_role,
        shift       as operator_home_shift
    from {{ ref('int_dim_operators') }}

),

enriched as (

    select
        -- Keys
        e.event_id,
        e.machine_id,

        -- Machine context
        m.machine_type,
        m.controller_type,
        m.location_cell,
        m.machine_age_years,
        m.age_band,
        m.is_aging_asset,

        -- Time
        e.event_timestamp,
        cast(e.event_timestamp as date)             as event_date,
        date_trunc('month', e.event_timestamp)      as event_month,
        e.shift,

        -- State
        e.machine_state,
        e.state_duration_minutes,
        e.spindle_utilization_pct,
        e.alarm_code,

        -- Operator context
        e.operator_empid,
        o.operator_id,
        o.operator_name,
        o.operator_role,

        -- Derived state flags
        (e.machine_state = 'RUNNING')                               as is_running,
        (e.machine_state = 'UNPLANNED_DOWN')                        as is_unplanned_down,
        (e.machine_state = 'PLANNED_DOWN')                          as is_planned_down,
        (e.machine_state = 'ALARM')                                 as is_alarm,
        (e.machine_state in ('UNPLANNED_DOWN', 'PLANNED_DOWN'))     as is_downtime

    from events e
    left join machines  m on e.machine_id     = m.machine_id
    left join operators o on e.operator_empid = o.employee_number

)

select * from enriched

-- ====================================================================
-- model: int_fct_maintenance_events
-- ====================================================================

-- Maintenance event fact enriched with the machine dimension and PM-compliance
-- flags. Grain: one row per maintenance_id. Dimensions are LEFT joined; join
-- resolution is asserted by not_null tests. PM compliance is derived here so
-- downstream marts share one definition.

with maint as (

    select * from {{ ref('stg_cmms__maintenance_records') }}

),

machines as (

    select
        machine_id,
        machine_type,
        controller_type,
        location_cell,
        machine_age_years,
        is_aging_asset
    from {{ ref('int_dim_machines') }}

),

technicians as (

    select
        employee_number,
        operator_id     as technician_id,
        operator_name   as technician_name
    from {{ ref('int_dim_operators') }}

),

enriched as (

    select
        -- Keys
        mt.maintenance_id,
        mt.machine_id,

        -- Machine context
        mc.machine_type,
        mc.controller_type,
        mc.location_cell,
        mc.machine_age_years,
        mc.is_aging_asset,

        -- Event detail
        mt.maintenance_type,
        mt.failure_code,
        mt.work_order_open_date,
        mt.work_order_close_date,
        cast(mt.work_order_open_date as date)           as event_date,
        date_trunc('month', mt.work_order_open_date)    as event_month,
        mt.downtime_hours,
        mt.parts_consumed,
        mt.resolution_notes,

        -- Technician context
        mt.technician_empid,
        t.technician_id,
        t.technician_name,

        -- PM schedule
        mt.pm_scheduled_date,
        mt.pm_completed_date,
        mt.days_overdue,

        -- Derived type and compliance flags
        (mt.maintenance_type = 'PLANNED_PM')                            as is_planned_pm,
        (mt.maintenance_type = 'UNPLANNED_REPAIR')                      as is_unplanned_repair,
        (mt.maintenance_type = 'INSPECTION')                            as is_inspection,
        (mt.maintenance_type = 'PLANNED_PM'
            and mt.pm_scheduled_date is null)                           as is_adhoc_pm,
        (mt.maintenance_type = 'PLANNED_PM'
            and mt.days_overdue is not null
            and mt.days_overdue <= 0)                                   as is_pm_ontime,
        (mt.maintenance_type = 'PLANNED_PM'
            and mt.days_overdue is not null
            and mt.days_overdue > 0)                                    as is_pm_late,
        (mt.maintenance_type = 'PLANNED_PM'
            and mt.days_overdue is not null
            and mt.days_overdue > {{ var('pm_overdue_threshold_days') }}) as is_pm_overdue_beyond_threshold

    from maint mt
    left join machines    mc on mt.machine_id       = mc.machine_id
    left join technicians t  on mt.technician_empid = t.employee_number

)

select * from enriched

-- ########################################################################
-- MARTS LAYER
-- ########################################################################

-- ====================================================================
-- model: mart_ml__rul_features
-- ====================================================================

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

-- ====================================================================
-- model: mart_oee__downtime_analysis
-- ====================================================================

-- Unified unplanned-downtime fact. Grain: one row per downtime event.
-- Two source perspectives, distinguished by source_system:
--   MACHINE_STATE  unplanned-down intervals from MachineMetrics, carrying
--                  within-shift time-of-day (feeds the shift-transition analysis)
--   CMMS_REPAIR    unplanned repairs from Limble, carrying failure_code and a
--                  valued downtime cost (feeds the failure-code Pareto and the
--                  cost-of-downtime summary)
-- Filter by source_system for each analysis; do not sum minutes across both.

with states as (

    select
        'ST-' || event_id                                       as downtime_key,
        'MACHINE_STATE'                                         as source_system,
        machine_id,
        machine_type,
        controller_type,
        location_cell,
        machine_age_years,
        is_aging_asset,
        event_timestamp,
        event_date,
        event_month,
        shift,
        cast(extract('hour' from event_timestamp) as integer)  as hour_of_day,
        cast(
            (extract('hour' from event_timestamp) * 60 + extract('minute' from event_timestamp))
            - (case when shift = 'A' then 6 * 60 else 14 * 60 end)
        as bigint)                                              as minute_of_shift,
        cast(null as varchar)                                   as failure_code,
        cast(state_duration_minutes as double)                 as downtime_minutes,
        round(state_duration_minutes / 60.0, 3)                as downtime_hours
    from {{ ref('int_fct_machine_states') }}
    where machine_state = 'UNPLANNED_DOWN'

),

repairs as (

    select
        'MN-' || maintenance_id                                 as downtime_key,
        'CMMS_REPAIR'                                           as source_system,
        machine_id,
        machine_type,
        controller_type,
        location_cell,
        machine_age_years,
        is_aging_asset,
        work_order_open_date                                    as event_timestamp,
        event_date,
        event_month,
        case when extract('hour' from work_order_open_date) < 14 then 'A' else 'B' end as shift,
        cast(extract('hour' from work_order_open_date) as integer) as hour_of_day,
        cast(null as bigint)                                    as minute_of_shift,
        failure_code,
        cast(downtime_hours * 60 as double)                     as downtime_minutes,
        downtime_hours
    from {{ ref('int_fct_maintenance_events') }}
    where is_unplanned_repair

),

unioned as (

    select * from states
    union all
    select * from repairs

),

final as (

    select
        downtime_key,
        source_system,
        machine_id,
        machine_type,
        controller_type,
        location_cell,
        machine_age_years,
        is_aging_asset,
        event_timestamp,
        event_date,
        event_month,
        shift,
        hour_of_day,
        minute_of_shift,
        (source_system = 'MACHINE_STATE'
            and minute_of_shift >= 0
            and minute_of_shift < {{ var('shift_startup_window_minutes') }})
                                                                as is_shift_startup,
        failure_code,
        downtime_minutes,
        downtime_hours,
        round(downtime_hours * case machine_type
            when 'CNC Lathe'       then {{ var('contribution_margin_by_type')['CNC Lathe'] }}
            when 'Vertical Mill'   then {{ var('contribution_margin_by_type')['Vertical Mill'] }}
            when 'Horizontal Mill' then {{ var('contribution_margin_by_type')['Horizontal Mill'] }}
        end, 2)                                                 as downtime_cost

    from unioned

)

select * from final

-- ====================================================================
-- model: mart_oee__machine_performance
-- ====================================================================

-- OEE decomposed into Availability, Performance, and Quality Rate.
-- Grain: one row per (machine_id, shift, day). A period_month column is carried
-- for monthly rollups. Metrics are computed per row and aggregated by summing
-- the underlying minutes downstream, never by averaging per-row OEE.
--
-- Availability  = run time / planned production time (scheduled - planned down)
-- Performance   = mean spindle utilisation while running (proxy; no cycle counts)
-- Quality Rate  = plant assumption (quality is tracked on paper, not digitally)
-- OEE           = Availability x Performance x Quality Rate

with states as (

    select * from {{ ref('int_fct_machine_states') }}

),

agg as (

    select
        machine_id,
        machine_type,
        controller_type,
        location_cell,
        machine_age_years,
        age_band,
        is_aging_asset,
        shift,
        event_date                                                              as period_date,
        event_month                                                             as period_month,

        -- filtered sums coalesced to 0: a state absent from a group contributes
        -- zero minutes rather than NULL (which would null out the arithmetic).
        sum(state_duration_minutes)                                                          as scheduled_minutes,
        coalesce(sum(state_duration_minutes) filter (where machine_state = 'PLANNED_DOWN'), 0)    as planned_down_minutes,
        coalesce(sum(state_duration_minutes) filter (where machine_state = 'RUNNING'), 0)         as run_minutes,
        coalesce(sum(state_duration_minutes) filter (where machine_state = 'UNPLANNED_DOWN'), 0)  as unplanned_down_minutes,
        coalesce(sum(state_duration_minutes) filter (where machine_state = 'SETUP'), 0)           as setup_minutes,
        coalesce(sum(state_duration_minutes) filter (where machine_state = 'IDLE'), 0)            as idle_minutes,
        count(*)                     filter (where machine_state = 'ALARM')                       as alarm_events,
        count(*)                     filter (where machine_state = 'UNPLANNED_DOWN')              as unplanned_down_events,

        coalesce(sum(spindle_utilization_pct * state_duration_minutes)
            filter (where machine_state = 'RUNNING' and spindle_utilization_pct is not null), 0)  as spindle_weighted,
        coalesce(sum(state_duration_minutes)
            filter (where machine_state = 'RUNNING' and spindle_utilization_pct is not null), 0)  as spindle_minutes

    from states
    group by 1, 2, 3, 4, 5, 6, 7, 8, 9, 10

),

final as (

    select
        {{ dbt_utils.generate_surrogate_key(['machine_id', 'shift', 'period_date']) }}  as performance_key,
        machine_id,
        machine_type,
        controller_type,
        location_cell,
        machine_age_years,
        age_band,
        is_aging_asset,
        shift,
        period_date,
        period_month,

        scheduled_minutes,
        planned_down_minutes,
        (scheduled_minutes - planned_down_minutes)      as planned_production_minutes,
        run_minutes,
        setup_minutes,
        idle_minutes,
        unplanned_down_minutes,
        unplanned_down_events,
        alarm_events,

        -- Guards handle daily edge cases (a shift-day with no planned production
        -- time or no running time), which cannot occur at coarser grains.
        case when (scheduled_minutes - planned_down_minutes) > 0
             then round(run_minutes::double / (scheduled_minutes - planned_down_minutes), 4)
             else 0 end                                                         as availability,

        case when spindle_minutes > 0
             then round((spindle_weighted / spindle_minutes) / 100.0, 4)
             else 0 end                                                         as performance,

        cast({{ var('assumed_quality_rate') }} as double)                       as quality_rate,

        case machine_type
            when 'CNC Lathe'       then {{ var('contribution_margin_by_type')['CNC Lathe'] }}
            when 'Vertical Mill'   then {{ var('contribution_margin_by_type')['Vertical Mill'] }}
            when 'Horizontal Mill' then {{ var('contribution_margin_by_type')['Horizontal Mill'] }}
        end                                                                     as contribution_margin_per_hour,

        round(
            (case when (scheduled_minutes - planned_down_minutes) > 0
                  then run_minutes::double / (scheduled_minutes - planned_down_minutes) else 0 end)
            * (case when spindle_minutes > 0 then (spindle_weighted / spindle_minutes) / 100.0 else 0 end)
            * {{ var('assumed_quality_rate') }}
        , 4)                                                                    as oee

    from agg

)

select * from final

-- ====================================================================
-- model: mart_oee__operator_setup
-- ====================================================================

-- Operator setup-time summary. Grain: one row per machinist.
-- Setup hours are reconciled from the ERP payroll number back to the HR operator
-- record, then compared against the machinist cohort median. Supports the
-- setup-time (training and standardisation) section of the diagnostic report.

with work_orders as (

    select operator_empid, setup_hours_actual
    from {{ ref('stg_erp__work_orders') }}
    where setup_hours_actual is not null

),

operators as (

    select employee_number, operator_id, operator_name, role, cert_status
    from {{ ref('int_dim_operators') }}
    where role = 'Machinist'

),

cohort as (

    select median(setup_hours_actual) as cohort_median_setup_hours
    from work_orders

),

per_operator as (

    select
        operator_empid,
        count(*)                            as total_jobs,
        round(median(setup_hours_actual), 2) as median_setup_hours,
        round(avg(setup_hours_actual), 2)    as mean_setup_hours
    from work_orders
    group by 1

),

final as (

    select
        o.operator_id,
        o.operator_name,
        o.role,
        o.cert_status,
        p.operator_empid,
        p.total_jobs,
        p.median_setup_hours,
        p.mean_setup_hours,
        round(c.cohort_median_setup_hours, 2)                                   as cohort_median_setup_hours,
        round(p.median_setup_hours / nullif(c.cohort_median_setup_hours, 0), 2) as setup_ratio_vs_cohort

    from per_operator p
    inner join operators o on p.operator_empid = o.employee_number
    cross join cohort c

)

select * from final

-- ====================================================================
-- model: mart_oee__pm_compliance
-- ====================================================================

-- Preventive-maintenance compliance summary. Grain: one row per machine.
-- Combines PM history (compliance rate, overdue statistics) with the current
-- tracker status (last PM, next due, days until/since due) as of the current
-- moment, taken as the end of the observation window.

with pm as (

    select * from {{ ref('int_fct_maintenance_events') }}
    where is_planned_pm

),

machines as (

    select
        machine_id,
        machine_type,
        controller_type,
        location_cell,
        machine_age_years,
        age_band,
        is_aging_asset
    from {{ ref('int_dim_machines') }}

),

history as (

    select
        machine_id,
        count(*)                                                as total_pms,
        count(*) filter (where is_adhoc_pm)                     as adhoc_pms,
        count(*) filter (where is_pm_ontime)                    as ontime_pms,
        count(*) filter (where is_pm_late)                      as late_pms,
        count(*) filter (where is_pm_overdue_beyond_threshold)  as overdue_beyond_threshold_pms,
        round(
            100.0 * count(*) filter (where is_pm_ontime)
            / nullif(count(*) filter (where days_overdue is not null), 0)
        , 1)                                                    as pct_ontime,
        round(avg(days_overdue) filter (where is_pm_late), 1)   as avg_days_overdue_when_late,
        max(days_overdue)                                       as max_days_overdue,
        max(pm_completed_date)                                  as last_pm_completed_date
    from pm
    group by 1

),

final as (

    select
        m.machine_id,
        m.machine_type,
        m.controller_type,
        m.location_cell,
        m.machine_age_years,
        m.age_band,
        m.is_aging_asset,

        h.total_pms,
        h.adhoc_pms,
        h.ontime_pms,
        h.late_pms,
        h.overdue_beyond_threshold_pms,
        h.pct_ontime,
        h.avg_days_overdue_when_late,
        h.max_days_overdue,

        h.last_pm_completed_date,
        date_diff('day', h.last_pm_completed_date, date '{{ var("end_date") }}')  as days_since_last_pm,
        h.last_pm_completed_date + {{ var('pm_interval_days') }}                    as next_pm_due_date,
        date_diff('day', date '{{ var("end_date") }}',
                  h.last_pm_completed_date + {{ var('pm_interval_days') }})         as days_until_next_pm,

        case
            when h.last_pm_completed_date + {{ var('pm_interval_days') }}
                 < date '{{ var("end_date") }}'                                  then 'Overdue'
            when h.last_pm_completed_date + {{ var('pm_interval_days') }}
                 <= date '{{ var("end_date") }}' + 7                             then 'Due Soon'
            else 'On Track'
        end                                                                        as pm_status

    from machines m
    left join history h on m.machine_id = h.machine_id

)

select * from final
