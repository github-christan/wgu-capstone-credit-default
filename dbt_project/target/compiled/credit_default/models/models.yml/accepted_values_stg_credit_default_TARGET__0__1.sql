
    
    

with all_values as (

    select
        TARGET as value_field,
        count(*) as n_records

    from CREDIT_DEFAULT.MODEL.stg_credit_default
    group by TARGET

)

select *
from all_values
where value_field not in (
    '0','1'
)


