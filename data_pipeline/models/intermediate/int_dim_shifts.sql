-- Shift calendar. Grain: one row per (calendar_date, shift_code) across the
-- operating window (Monday-Saturday, two shifts). Provides explicit shift
-- windows so facts can attribute a timestamp to a shift. Attribution is by
-- start time: work that overruns the shift end (e.g. a job starting 14:00 that
-- runs past 22:00) remains attributed to the shift it started in.

with calendar as (

    select cast(unnest(generate_series(
        timestamp '{{ var("start_date") }}',
        timestamp '{{ var("end_date") }}',
        interval '1 day'
    )) as date)                                     as calendar_date

),

shifts as (

    select 'A' as shift_code, 'Shift A' as shift_label, 6  as start_hour, 14 as end_hour
    union all
    select 'B' as shift_code, 'Shift B' as shift_label, 14 as start_hour, 22 as end_hour

),

final as (

    select
        cast(c.calendar_date as varchar) || '-' || s.shift_code   as shift_key,
        c.calendar_date,
        s.shift_code,
        s.shift_label,
        c.calendar_date + s.start_hour * interval '1 hour'        as shift_start_ts,
        c.calendar_date + s.end_hour   * interval '1 hour'        as shift_end_ts,
        isodow(c.calendar_date)                                   as day_of_week,
        (isodow(c.calendar_date) = 6)                             as is_saturday,
        date_trunc('month', c.calendar_date)                      as calendar_month

    from calendar c
    cross join shifts s
    where isodow(c.calendar_date) <= 6   -- Monday-Saturday

)

select * from final
