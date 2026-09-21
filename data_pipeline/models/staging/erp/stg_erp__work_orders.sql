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
