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
