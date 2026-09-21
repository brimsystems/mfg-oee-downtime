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
