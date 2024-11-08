import pytest
import numpy as np


def test_nrb(mobilecapacity):
    # Perform calculations
    nrb_calc = mobilecapacity.nrb
    # Expected results (hardcoded for comparison)
    nrb_expected = 175
    # Compare Python calculation results with expected results
    assert nrb_calc == nrb_expected, f"Expected {nrb_expected} but got {nrb_calc}"


def test_avrbpdsch(mobilecapacity):
    # Perform calculations
    avrbpdsch_calc = round(mobilecapacity.avrbpdsch, 2)
    # Expected results (hardcoded for comparison)
    avrbpdsch_expected = 46.86
    # Compare Python calculation results with expected results
    assert avrbpdsch_calc == avrbpdsch_expected, f"Expected {avrbpdsch_expected} but got {avrbpdsch_calc}"


# @pytest.mark.parametrize("d, expected", [
#     (7000, np.inf),
#     (2000, 13.67)
# ])
# def test_poiddatareq(mobilecapacity, d, expected):
#     # Perform calculation
#     poiddatareq_calc = round(mobilecapacity.poiddatareq(d=d)[0], 2) \
#         if expected is not None else mobilecapacity.poiddatareq(d=d)[0]

#     # Compare calculation results with expected results
#     assert poiddatareq_calc == expected, f"Expected {expected} but got {poiddatareq_calc}"


# def test_brrbpopcd(mobilecapacity, popcd=5000):
#     # Perform calculation
#     brrbpopcd_calc = round(mobilecapacity.brrbpopcd(popcd)[0], 2)

#     # Expected results (hardcoded for comparison)
#     brrbpopcd_expected = 683.97

#     # Compare calculation results with expected results
#     assert brrbpopcd_calc == brrbpopcd_expected, f"Expected {brrbpopcd_expected} but got {brrbpopcd_calc}"


def test_avubrnonbh(mobilecapacity, udatavmonth=5):
    # Perform calculation
    avubrnonbh_calc = round(mobilecapacity.avubrnonbh(udatavmonth), 2)

    # Expected results (hardcoded for comparison)
    avubrnonbh_expected = 19.62

    # Compare calculation results with expected results
    assert avubrnonbh_calc == avubrnonbh_expected, f"Expected {avubrnonbh_expected} but got {avubrnonbh_calc}"


def test_upopbr(mobilecapacity, avubrnonbh=19.62247, pop=10000):
    # Perform calculation
    upopbr_calc = round(mobilecapacity.upopbr(avubrnonbh, pop), 2)

    # Expected results (hardcoded for comparison)
    upopbr_expected = 36955.65

    # Compare calculation results with expected results
    assert upopbr_calc == upopbr_expected, f"Expected {upopbr_expected} but got {upopbr_calc}"


def test_upoprbu(mobilecapacity, upopbr=32704.12, brrbpopcd=2081.84):
    # Perform calculation
    upoprbu_calc = round(mobilecapacity.upoprbu(upopbr, brrbpopcd)[0], 2)

    # Expected results (hardcoded for comparison)
    upoprbu_expected = 15.71

    # Compare calculation results with expected results
    assert upoprbu_calc == upoprbu_expected, f"Expected {upoprbu_expected} but got {upoprbu_calc}"


def test_cellavcap(mobilecapacity, avrbpdsch=82, upoprbu=15.71):
    # Perform calculation
    cellavcap_calc = round(mobilecapacity.cellavcap(avrbpdsch, upoprbu)[0], 2)

    # Expected results (hardcoded for comparison)
    cellavcap_expected = 66.29

    # Compare calculation results with expected results
    assert cellavcap_calc == cellavcap_expected, f"Expected {cellavcap_expected} but got {cellavcap_calc}"


@pytest.mark.parametrize("cellavcap, rbdlthtarg, sufcapch_expected", [
    (9.29, 10.93, False),  # Test case 1
    (90, 85, True)    # Test case 2
])
def test_sufcapch(mobilecapacity, sufcapch_expected, cellavcap, rbdlthtarg):
    # Perform the calculation.
    sufcapch_calc = mobilecapacity.sufcapch(cellavcap, rbdlthtarg)

    # Compare the calculation results with the expected results.
    assert sufcapch_calc == sufcapch_expected, f"Expected {sufcapch_expected} but got {sufcapch_calc}"
