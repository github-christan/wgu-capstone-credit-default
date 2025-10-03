
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select TARGET
from CREDIT_DEFAULT.MODEL.stg_credit_default
where TARGET is null



  
  
      
    ) dbt_internal_test