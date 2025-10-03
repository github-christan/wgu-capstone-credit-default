
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    

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



  
  
      
    ) dbt_internal_test