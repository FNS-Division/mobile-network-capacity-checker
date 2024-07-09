from mobile_capacity.capacity import Capacity
import os
import pytest
import numpy as np

def test_nrb(init_variable_values):
    mobilecapacity = Capacity(**init_variable_values)
    # Perform calculations  
    nrb_calc = mobilecapacity.nrb
    # Expected results (hardcoded for comparison) 
    nrb_expected = 100 
    # Compare Python calculation results with expected results  
    assert nrb_calc == nrb_expected, f"Expected {nrb_expected} but got {nrb_calc}"

def test_avrbpdsch(init_variable_values):
    mobilecapacity = Capacity(**init_variable_values)
    # Perform calculations  
    avrbpdsch_calc = mobilecapacity.avrbpdsch
    # Expected results (hardcoded for comparison) 
    avrbpdsch_expected = 82 
    # Compare Python calculation results with expected results  
    assert avrbpdsch_calc == avrbpdsch_expected, f"Expected {avrbpdsch_expected} but got {avrbpdsch_calc}"

# @pytest.mark.parametrize("d, expected", [
#     (7000, np.inf),
#     (2000, 9.7)
# ])
# def test_poiddatareq(init_variable_values, d, expected):

#     mobilecapacity = Capacity(**init_variable_values)

#     # Perform calculation
#     poiddatareq_calc = round(mobilecapacity.poiddatareq(d=d), 2) \
#         if expected is not None else mobilecapacity.poiddatareq(d=d)

#     # Compare calculation results with expected results
#     assert poiddatareq_calc == expected, f"Expected {expected} but got {poiddatareq_calc}"

# def test_brrbpopcd(init_variable_values, popcd=5000):

#     mobilecapacity = Capacity(**init_variable_values)

#     # Perform calculation
#     brrbpopcd_calc = round(mobilecapacity.brrbpopcd(popcd), 2)
    
#     # Expected results (hardcoded for comparison) 
#     brrbpopcd_expected = 2081.84 

#     # Compare calculation results with expected results
#     assert brrbpopcd_calc == brrbpopcd_expected, f"Expected {brrbpopcd_expected} but got {brrbpopcd_calc}"

def test_avubrnonbh(init_variable_values, udatavmonth = 5):

    mobilecapacity = Capacity(**init_variable_values)

    # Perform calculation
    avubrnonbh_calc = round(mobilecapacity.avubrnonbh(udatavmonth), 2)
    
    # Expected results (hardcoded for comparison) 
    avubrnonbh_expected = 19.62 

    # Compare calculation results with expected results
    assert avubrnonbh_calc == avubrnonbh_expected, f"Expected {avubrnonbh_expected} but got {avubrnonbh_calc}"

def test_upopbr(init_variable_values, avubrnonbh = 19.62247, pop = 10000):

    mobilecapacity = Capacity(**init_variable_values)

    # Perform calculation
    upopbr_calc = round(mobilecapacity.upopbr(avubrnonbh, pop), 2)
    
    # Expected results (hardcoded for comparison) 
    upopbr_expected = 32704.12 

    # Compare calculation results with expected results
    assert upopbr_calc == upopbr_expected, f"Expected {upopbr_expected} but got {upopbr_calc}"

def test_upoprbu(init_variable_values, upopbr = 32704.12, brrbpopcd = 2081.84):

    mobilecapacity = Capacity(**init_variable_values)

    # Perform calculation
    upoprbu_calc = round(mobilecapacity.upoprbu(upopbr, brrbpopcd), 2)
    
    # Expected results (hardcoded for comparison) 
    upoprbu_expected = 15.71 

    # Compare calculation results with expected results
    assert upoprbu_calc == upoprbu_expected, f"Expected {upoprbu_expected} but got {upoprbu_calc}"

def test_cellavcap(init_variable_values, avrbpdsch = 82, upoprbu = 15.71):

    mobilecapacity = Capacity(**init_variable_values)

    # Perform calculation
    cellavcap_calc = round(mobilecapacity.cellavcap(avrbpdsch, upoprbu), 2)
    
    # Expected results (hardcoded for comparison) 
    cellavcap_expected = 66.29 

    # Compare calculation results with expected results
    assert cellavcap_calc == cellavcap_expected, f"Expected {cellavcap_expected} but got {cellavcap_calc}"
   
@pytest.mark.parametrize("cellavcap, rbdlthtarg, sufcapch_expected", [  
    (9.29, 10.93, False),  # Test case 1  
    (90, 85, True)    # Test case 2  
])   
def test_sufcapch(init_variable_values, sufcapch_expected, cellavcap, rbdlthtarg):  
  
    mobilecapacity = Capacity(**init_variable_values)  
  
    # Perform the calculation.  
    sufcapch_calc = mobilecapacity.sufcapch(cellavcap, rbdlthtarg)
  
    # Compare the calculation results with the expected results.  
    assert sufcapch_calc == sufcapch_expected, f"Expected {sufcapch_expected} but got {sufcapch_calc}"  