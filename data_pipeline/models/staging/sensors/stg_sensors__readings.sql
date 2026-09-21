-- Daily condition-monitoring readings from the retrofit IIoT gateway.
-- Grain: one row per (machine_id, reading_date).

with source as (

    select * from {{ source('sensors', 'sensor_readings') }}

),

staged as (

    select
        machine_id,
        cast(reading_date as date)              as reading_date,
        cast(vibration_rms_mm_s as double)      as vibration_rms_mm_s,
        cast(bearing_temp_c as double)          as bearing_temp_c,
        cast(spindle_power_kw as double)        as spindle_power_kw,
        cast(hydraulic_pressure_bar as double)  as hydraulic_pressure_bar

    from source

)

select * from staged
