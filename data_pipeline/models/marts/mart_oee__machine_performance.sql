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
