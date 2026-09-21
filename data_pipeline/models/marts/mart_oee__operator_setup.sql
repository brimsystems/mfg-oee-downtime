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
