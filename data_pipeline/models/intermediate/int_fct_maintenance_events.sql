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
