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
