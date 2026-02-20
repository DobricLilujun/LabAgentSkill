# =============================================================================
# Evaluation Utilities for LabAgentSkill
# =============================================================================
# This module centralizes all label-extraction and prediction-parsing functions
# used to evaluate agent/skill outputs across different benchmarks:
#   - Binary sentiment classification (IMDB): get_predicted_label()
#   - XBRL financial tag classification (FiNER-139): get_prediction_XBRL_TAGS()
#   - Insurance yes/no classification (InsurBench): get_insurBench_predicted_label()
#
# Each function takes a raw LLM response string and returns a normalized label
# that can be directly compared against the ground-truth for accuracy computation.
# =============================================================================

import re


# -------------------------------------------------------------------------
# Sentiment Classification (IMDB)
# -------------------------------------------------------------------------

def get_predicted_label(message_classification: str) -> str:
    """
    Extract a binary sentiment label from an LLM response for the IMDB task.

    Uses a series of heuristic rules applied in priority order:
      1. If "positive" appears and "negative" does NOT → "positive"
      2. If "negative" appears and "positive" does NOT → "negative"
      3. If the response starts with "positive" or "negative" → use that
         (handles ambiguous cases where both words appear but one leads)
      4. Otherwise → "unknown"

    Args:
        message_classification: Raw text response from the agent/LLM.

    Returns:
        One of "positive", "negative", or "unknown".
    """
    # Normalize: handle None, strip whitespace, and lowercase for comparison
    msg_lower = (message_classification or "").strip().lower()

    # Unambiguous cases: only one sentiment keyword present
    if "positive" in msg_lower and "negative" not in msg_lower:
        return "positive"
    if "negative" in msg_lower and "positive" not in msg_lower:
        return "negative"

    # Ambiguous case: both keywords present — use whichever appears first
    if msg_lower.startswith("positive"):
        return "positive"
    if msg_lower.startswith("negative"):
        return "negative"

    # Could not determine sentiment
    return "unknown"



# -------------------------------------------------------------------------
# XBRL Tag Classification (FiNER-139)
# -------------------------------------------------------------------------
# Complete list of 139 XBRL (eXtensible Business Reporting Language) taxonomy
# tags used in the FiNER-139 financial NER dataset. These tags represent
# standardized financial concepts (e.g., revenue, debt, stock compensation)
# that numeric entities in SEC filings are mapped to.

XBRL_TAGS = [
    "SharebasedCompensationArrangementBySharebasedPaymentAwardAwardVestingRightsPercentage",
    "InterestExpense",
    "GoodwillImpairmentLoss",
    "SaleOfStockPricePerShare",
    "BusinessCombinationAcquisitionRelatedCosts",
    "LineOfCreditFacilityCurrentBorrowingCapacity",
    "LineOfCreditFacilityMaximumBorrowingCapacity",
    "PreferredStockSharesAuthorized",
    "RestructuringCharges",
    "IncomeLossFromEquityMethodInvestments",
    "EquityMethodInvestmentOwnershipPercentage",
    "Revenues",
    "NumberOfRealEstateProperties",
    "CumulativeEffectOfNewAccountingPrincipleInPeriodOfAdoption",
    "IncomeTaxExpenseBenefit",
    "SharebasedCompensationArrangementBySharebasedPaymentAwardExpirationPeriod",
    "DebtInstrumentFairValue",
    "AccrualForEnvironmentalLossContingencies",
    "CommonStockDividendsPerShareDeclared",
    "UnrecognizedTaxBenefitsThatWouldImpactEffectiveTaxRate",
    "Goodwill",
    "CommonStockSharesAuthorized",
    "UnrecognizedTaxBenefits",
    "LineOfCredit",
    "PublicUtilitiesRequestedRateIncreaseDecreaseAmount",
    "EquityMethodInvestments",
    "LineOfCreditFacilityUnusedCapacityCommitmentFeePercentage",
    "ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsVestedInPeriodTotalFairValue",
    "CommonStockCapitalSharesReservedForFutureIssuance",
    "DebtInstrumentConvertibleConversionPrice1",
    "LossContingencyPendingClaimsNumber",
    "OperatingLeasePayments",
    "LongTermDebtFairValue",
    "LeaseAndRentalExpense",
    "OperatingLeaseWeightedAverageRemainingLeaseTerm1",
    "LongTermDebt",
    "ClassOfWarrantOrRightExercisePriceOfWarrantsOrRights1",
    "DefinedContributionPlanCostRecognized",
    "LesseeOperatingLeaseTermOfContract",
    "ShareBasedCompensationArrangementByShareBasedPaymentAwardNumberOfSharesAuthorized",
    "DebtWeightedAverageInterestRate",
    "GuaranteeObligationsMaximumExposure",
    "DebtInstrumentTerm",
    "CapitalizedContractCostAmortization",
    "FiniteLivedIntangibleAssetUsefulLife",
    "ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsGrantsInPeriodGross",
    "DebtInstrumentInterestRateEffectivePercentage",
    "LettersOfCreditOutstandingAmount",
    "NumberOfOperatingSegments",
    "AllocatedShareBasedCompensationExpense",
    "CashAndCashEquivalentsFairValueDisclosure",
    "ContractWithCustomerLiabilityRevenueRecognized",
    "EmployeeServiceShareBasedCompensationTaxBenefitFromCompensationExpense",
    "LineOfCreditFacilityCommitmentFeePercentage",
    "DerivativeNotionalAmount",
    "AntidilutiveSecuritiesExcludedFromComputationOfEarningsPerShareAmount",
    "TreasuryStockAcquiredAverageCostPerShare",
    "RevenueFromRelatedParties",
    "BusinessAcquisitionPercentageOfVotingInterestsAcquired",
    "AmortizationOfIntangibleAssets",
    "BusinessCombinationRecognizedIdentifiableAssetsAcquiredAndLiabilitiesAssumedIntangibleAssetsOtherThanGoodwill",
    "ContractWithCustomerLiability",
    "ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsGrantsInPeriodWeightedAverageGrantDateFairValue",
    "AssetImpairmentCharges",
    "DebtInstrumentBasisSpreadOnVariableRate1",
    "BusinessCombinationConsiderationTransferred1",
    "DebtInstrumentUnamortizedDiscount",
    "PaymentsToAcquireBusinessesNetOfCashAcquired",
    "ShareBasedCompensationArrangementByShareBasedPaymentAwardAwardVestingPeriod1",
    "DebtInstrumentCarryingAmount",
    "AcquiredFiniteLivedIntangibleAssetsWeightedAverageUsefulLife",
    "DerivativeFixedInterestRate",
    "ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsGrantsInPeriod",
    "TreasuryStockValueAcquiredCostMethod",
    "OperatingLossCarryforwards",
    "DebtInstrumentMaturityDate",
    "ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsNonvestedNumber",
    "DefinedBenefitPlanContributionsByEmployer",
    "GainsLossesOnExtinguishmentOfDebt",
    "AreaOfRealEstateProperty",
    "BusinessAcquisitionEquityInterestsIssuedOrIssuableNumberOfSharesIssued",
    "SaleOfStockNumberOfSharesIssuedInTransaction",
    "SupplementalInformationForPropertyCasualtyInsuranceUnderwritersPriorYearClaimsAndClaimsAdjustmentExpense",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "DeferredFinanceCostsGross",
    "NumberOfReportableSegments",
    "BusinessCombinationContingentConsiderationLiability",
    "ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsGrantsInPeriodWeightedAverageGrantDateFairValue",
    "RepaymentsOfDebt",
    "SharePrice",
    "ShareBasedCompensationArrangementByShareBasedPaymentAwardNumberOfSharesAvailableForGrant",
    "StockRepurchaseProgramAuthorizedAmount1",
    "LineOfCreditFacilityRemainingBorrowingCapacity",
    "PropertyPlantAndEquipmentUsefulLife",
    "ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsExercisesInPeriodTotalIntrinsicValue",
    "DisposalGroupIncludingDiscontinuedOperationConsideration",
    "DebtInstrumentRedemptionPricePercentage",
    "DebtInstrumentInterestRateStatedPercentage",
    "OperatingLeasesRentExpenseNet",
    "StockRepurchaseProgramRemainingAuthorizedRepurchaseAmount1",
    "AmortizationOfFinancingCosts",
    "ConcentrationRiskPercentage1",
    "Depreciation",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RelatedPartyTransactionExpensesFromTransactionsWithRelatedParty",
    "DebtInstrumentFaceAmount",
    "RestructuringAndRelatedCostExpectedCost1",
    "EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognizedPeriodForRecognition1",
    "MinorityInterestOwnershipPercentageByNoncontrollingOwners",
    "CommonStockParOrStatedValuePerShare",
    "MinorityInterestOwnershipPercentageByParent",
    "CommonStockSharesOutstanding",
    "EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognized",
    "DeferredFinanceCostsNet",
    "ShareBasedCompensation",
    "InterestExpenseDebt",
    "StockIssuedDuringPeriodSharesNewIssues",
    "EffectiveIncomeTaxRateContinuingOperations",
    "BusinessCombinationRecognizedIdentifiableAssetsAcquiredAndLiabilitiesAssumedIntangibles",
    "OperatingLeaseExpense",
    "PreferredStockDividendRatePercentage",
    "StockRepurchasedDuringPeriodShares",
    "OperatingLeaseCost",
    "ProceedsFromIssuanceOfCommonStock",
    "StockRepurchasedAndRetiredDuringPeriodShares",
    "RelatedPartyTransactionAmountsOfTransaction",
    "EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognizedShareBasedAwardsOtherThanOptions",
    "OperatingLeaseLiability",
    "EffectiveIncomeTaxRateReconciliationAtFederalStatutoryIncomeTaxRate",
    "OperatingLeaseWeightedAverageDiscountRatePercent",
    "PaymentsToAcquireBusinessesGross",
    "LossContingencyDamagesSoughtValue",
    "TreasuryStockSharesAcquired",
    "LossContingencyAccrualAtCarryingValue",
    "RevenueRemainingPerformanceObligation",
    "LineOfCreditFacilityInterestRateAtPeriodEnd",
    "LesseeOperatingLeaseRenewalTerm",
    "OperatingLeaseRightOfUseAsset",
    "LossContingencyEstimateOfPossibleLoss",
]  # 139 tags total

# Pre-build a case-insensitive lookup dictionary.
# Maps each lowercased tag to its original CamelCase form, enabling
# efficient case-insensitive substring searches in get_prediction_XBRL_TAGS().
_XBRL_TAGS_LOWER = {tag.lower(): tag for tag in XBRL_TAGS}


def get_prediction_XBRL_TAGS(message: str) -> str:
    """
    Extract a predicted XBRL tag from a model response string.

    Matching strategy (applied in order of priority):
        1. Exact (case-sensitive) substring match across all 139 tags.
        2. Case-insensitive substring match as a fallback.

    In both stages, if multiple tags match (e.g., "Goodwill" and
    "GoodwillImpairmentLoss" both appear), the **longest** match is
    returned to avoid partial/short false positives.

    Args:
        message: The raw response string from the agent / LLM.

    Returns:
        The matched XBRL tag name in its original CamelCase form,
        or ``"unknown"`` if no tag could be identified.
    """
    if not message:
        return "unknown"

    text = message.strip()

    # --- Stage 1: Exact (case-sensitive) substring match ---
    # Collect all tags that appear verbatim in the response text
    exact_matches = [tag for tag in XBRL_TAGS if tag in text]
    if exact_matches:
        # Return the longest match to avoid partial hits
        return max(exact_matches, key=len)

    # --- Stage 2: Case-insensitive substring match (fallback) ---
    # Useful when the LLM returns the tag in a different casing
    text_lower = text.lower()
    ci_matches = [
        original
        for lower, original in _XBRL_TAGS_LOWER.items()
        if lower in text_lower
    ]
    if ci_matches:
        return max(ci_matches, key=len)

    # No known XBRL tag found in the response
    return "unknown"



# -------------------------------------------------------------------------
# Insurance Yes/No Classification (InsurBench)
# -------------------------------------------------------------------------

# Pre-compiled word-boundary regex patterns for strict "yes" / "no" matching.
# Word boundaries (\b) prevent false positives like "yesterday" or "ノート".
_yes = re.compile(r"\byes\b", re.IGNORECASE)
_no  = re.compile(r"\bno\b",  re.IGNORECASE)


def get_insurBench_predicted_label(message_classification: str) -> str:
    """
    Extract a yes/no label from an LLM response for the InsurBench task.

    Uses word-boundary regex to match whole words only, which avoids
    false positives from substrings (e.g., "yesterday" won't match "yes").
    If both "yes" and "no" appear, "YES" takes priority.

    Args:
        message_classification: Raw text response from the agent/LLM.

    Returns:
        "YES", "NO", or "unknown".
    """
    s = message_classification.strip()
    if _yes.search(s):
        return "YES"
    if _no.search(s):
        return "NO"
    return "unknown"


def get_insurBench_predicted_label_v2(message_classification: str) -> str:
    """
    Simplified (v2) yes/no label extractor for InsurBench.

    Unlike v1, this uses plain substring matching instead of word-boundary
    regex. This is more lenient — it will match "yesterday" as "YES" and
    "notion" as "NO". Use v1 when precision matters; use v2 when recall
    is more important or responses are well-structured.

    Args:
        message_classification: Raw text response from the agent/LLM.

    Returns:
        "YES", "NO", or "unknown".
    """
    s = message_classification.strip().lower()
    if "yes" in s:
        return "YES"
    if "no" in s:
        return "NO"
    return "unknown"