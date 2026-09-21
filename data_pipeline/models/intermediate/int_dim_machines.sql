-- Conformed machine dimension. Grain: one row per machine_id.
-- machine_id is consistent across the MachineMetrics, ERP, and CMMS extracts,
-- so this is a clean reference with a few derived analytic attributes.

with source as (

    select * from {{ ref('stg_machinemetrics__machines') }}

),

final as (

    select
        machine_id,
        machine_type,
        controller_type,
        location_cell,
        machine_age_years,
        max_spindle_rpm,
        install_date,

        case
            when machine_age_years < 3  then '<3 yrs'
            when machine_age_years <= 7 then '3-7 yrs'
            when machine_age_years <= 10 then '7-10 yrs'
            else '11+ yrs'
        end                                 as age_band,

        (machine_age_years > 9)             as is_aging_asset

    from source

)

select * from final
