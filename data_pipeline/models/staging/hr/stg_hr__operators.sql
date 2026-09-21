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
