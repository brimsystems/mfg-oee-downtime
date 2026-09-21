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
