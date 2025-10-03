
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select LIMIT_BAL
from CREDIT_DEFAULT.MODEL.feature_view_credit_default
where LIMIT_BAL is null



  
  
      
    ) dbt_internal_test