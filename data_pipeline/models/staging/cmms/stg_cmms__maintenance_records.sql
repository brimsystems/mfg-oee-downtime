with source as (

    select * from {{ source('cmms', 'maintenance_records') }}

),

staged as (

    select
        maintenance_id,
        machine_id,
        maintenance_type,

        -- Populated for unplanned repairs only; null for PM and inspections.
        failure_code,

        cast(work_order_open_date as timestamp)     as work_order_open_date,
        cast(work_order_close_date as timestamp)    as work_order_close_date,
        cast(downtime_hours as double)              as downtime_hours,
        cast(technician_empid as integer)           as technician_empid,
        parts_consumed,
        resolution_notes,

        -- Ad-hoc PMs are logged with a completion but no scheduled date, so the
        -- scheduled date and days_overdue are null for those rows. days_overdue
        -- is retained from source as (completed - scheduled); negative is early.
        cast(pm_scheduled_date as date)             as pm_scheduled_date,
        cast(pm_completed_date as date)             as pm_completed_date,
        cast(days_overdue as integer)               as days_overdue

    from source

)

select * from staged
