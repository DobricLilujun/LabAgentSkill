# All Related evaluation method for LabAgentSkill should be placed here. 
# For example, if you want to evaluate the accuracy of a classification skill, you can implement a function like `get_predicted_label` to extract the predicted label from the skill's output, and then compare it with the true label to calculate accuracy metrics.
import re



def get_predicted_label(message_classification: str) -> str:
    """Determine predicted label from a classification string."""
    msg_lower = (message_classification or "").strip().lower()

    if "positive" in msg_lower and "negative" not in msg_lower:
        return "positive"
    if "negative" in msg_lower and "positive" not in msg_lower:
        return "negative"
    if msg_lower.startswith("positive"):
        return "positive"
    if msg_lower.startswith("negative"):
        return "negative"
    return "unknown"



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
]  # 139 tags

# Pre-build a case-insensitive lookup: lowercase tag -> original tag
_XBRL_TAGS_LOWER = {tag.lower(): tag for tag in XBRL_TAGS}


def get_prediction_XBRL_TAGS(message: str) -> str:
    """Extract a predicted XBRL tag from a model response string.

    Matching strategy (applied in order):
        1. Exact match — scan for any known tag appearing verbatim in the text.
        2. Case-insensitive match — same scan but ignoring case.
        3. Longest-substring match — if multiple tags match, prefer the longest
           one to avoid partial hits (e.g. "Goodwill" vs "GoodwillImpairmentLoss").

    Args:
        message: The raw response string from the agent / LLM.

    Returns:
        The matched XBRL tag name (original casing), or ``"unknown"`` if no
        tag could be identified.
    """
    if not message:
        return "unknown"

    text = message.strip()

    # --- 1. Exact (case-sensitive) match — prefer longest ---
    exact_matches = [tag for tag in XBRL_TAGS if tag in text]
    if exact_matches:
        return max(exact_matches, key=len)

    # --- 2. Case-insensitive match — prefer longest ---
    text_lower = text.lower()
    ci_matches = [
        original
        for lower, original in _XBRL_TAGS_LOWER.items()
        if lower in text_lower
    ]
    if ci_matches:
        return max(ci_matches, key=len)

    return "unknown"



_yes = re.compile(r"\byes\b", re.IGNORECASE)
_no  = re.compile(r"\bno\b",  re.IGNORECASE)

def get_insurBench_predicted_label(message_classification: str) -> str:
    s = message_classification.strip()
    if _yes.search(s):
        return "YES"
    if _no.search(s):
        return "NO"
    return "unknown"