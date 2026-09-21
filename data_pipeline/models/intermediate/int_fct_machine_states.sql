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
