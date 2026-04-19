"""
Heads Up For Tails (HUFT) — Supply Chain & Marketing Dataset Generator
=======================================================================
Generates realistic synthetic data modelled on HUFT's actual business:
  - 67 stores across 4 regions (60 physical + 2 online + 5 spa)
  - 65 SKUs from real pet store brands (Royal Canin, Pedigree, Sara's, Farmina,
    Drools, Whiskas, KONG, Trixie, Virbac, Ruffwear + HUFT private labels)
  - Indian INR pricing
  - India-specific seasonality: Diwali, monsoon, summer, New Year
  - Cold-chain SKUs (Sara's fresh food)
  - Multi-channel: online + offline stores + HUFT Spa
  - Customer segments, promotions, returns, brand performance

Output files (all in data/):
  huft_daily_demand.csv          — core daily demand/inventory (65 SKUs × 730 days)
  huft_stores.csv                — 120 store master
  huft_products.csv              — 80 product master with full HUFT attributes
  huft_customers.csv             — 5000 customer records with segments
  huft_promotions.csv            — 48 marketing campaigns (2023-2024)
  huft_sales_transactions.csv    — 50,000 individual sales transactions
  huft_returns.csv               — return log (~3% return rate)
  huft_supplier_performance.csv  — monthly supplier scorecard
  huft_cold_chain.csv            — cold-chain temperature/expiry log for fresh SKUs
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
rng = np.random.default_rng(SEED)

OUT = Path(__file__).parent

# ── Indian cities and store network ──────────────────────────────────────────

STORES = [
    # format: store_id, city, state, region, store_type, opened_year, size_sqft, has_spa
    ("ST001", "Mumbai", "Maharashtra", "West", "Flagship", 2013, 3500, True),
    ("ST002", "Mumbai", "Maharashtra", "West", "Standard", 2015, 1800, False),
    ("ST003", "Mumbai", "Maharashtra", "West", "Standard", 2017, 1600, True),
    ("ST004", "Mumbai", "Maharashtra", "West", "Express", 2019, 900, False),
    ("ST005", "Mumbai", "Maharashtra", "West", "Standard", 2020, 1700, False),
    ("ST006", "Pune", "Maharashtra", "West", "Standard", 2016, 2000, True),
    ("ST007", "Pune", "Maharashtra", "West", "Standard", 2018, 1500, False),
    ("ST008", "Pune", "Maharashtra", "West", "Express", 2021, 850, False),
    ("ST009", "Ahmedabad", "Gujarat", "West", "Standard", 2017, 1800, False),
    ("ST010", "Ahmedabad", "Gujarat", "West", "Standard", 2020, 1600, False),
    ("ST011", "Surat", "Gujarat", "West", "Express", 2021, 900, False),
    ("ST012", "Vadodara", "Gujarat", "West", "Express", 2022, 850, False),
    ("ST013", "New Delhi", "Delhi", "North", "Flagship", 2013, 4000, True),
    ("ST014", "New Delhi", "Delhi", "North", "Standard", 2014, 2200, True),
    ("ST015", "New Delhi", "Delhi", "North", "Standard", 2016, 1800, False),
    ("ST016", "New Delhi", "Delhi", "North", "Standard", 2018, 1700, False),
    ("ST017", "New Delhi", "Delhi", "North", "Express", 2019, 900, False),
    ("ST018", "Gurgaon", "Haryana", "North", "Flagship", 2015, 3200, True),
    ("ST019", "Gurgaon", "Haryana", "North", "Standard", 2017, 1900, True),
    ("ST020", "Gurgaon", "Haryana", "North", "Express", 2020, 900, False),
    ("ST021", "Noida", "UP", "North", "Standard", 2016, 1800, False),
    ("ST022", "Noida", "UP", "North", "Standard", 2018, 1600, False),
    ("ST023", "Noida", "UP", "North", "Express", 2021, 850, False),
    ("ST024", "Chandigarh", "Punjab", "North", "Standard", 2017, 1700, False),
    ("ST025", "Chandigarh", "Punjab", "North", "Express", 2020, 900, False),
    ("ST026", "Jaipur", "Rajasthan", "North", "Standard", 2018, 1700, False),
    ("ST027", "Jaipur", "Rajasthan", "North", "Express", 2021, 900, False),
    ("ST028", "Lucknow", "UP", "North", "Standard", 2019, 1600, False),
    ("ST029", "Ludhiana", "Punjab", "North", "Express", 2022, 850, False),
    ("ST030", "Dehradun", "Uttarakhand", "North", "Express", 2022, 800, False),
    ("ST031", "Bengaluru", "Karnataka", "South", "Flagship", 2014, 4200, True),
    ("ST032", "Bengaluru", "Karnataka", "South", "Standard", 2015, 2400, True),
    ("ST033", "Bengaluru", "Karnataka", "South", "Standard", 2016, 2000, True),
    ("ST034", "Bengaluru", "Karnataka", "South", "Standard", 2017, 1900, False),
    ("ST035", "Bengaluru", "Karnataka", "South", "Standard", 2018, 1800, False),
    ("ST036", "Bengaluru", "Karnataka", "South", "Express", 2019, 950, False),
    ("ST037", "Bengaluru", "Karnataka", "South", "Express", 2020, 900, False),
    ("ST038", "Hyderabad", "Telangana", "South", "Flagship", 2015, 3800, True),
    ("ST039", "Hyderabad", "Telangana", "South", "Standard", 2016, 2100, True),
    ("ST040", "Hyderabad", "Telangana", "South", "Standard", 2018, 1800, False),
    ("ST041", "Hyderabad", "Telangana", "South", "Express", 2020, 900, False),
    ("ST042", "Chennai", "Tamil Nadu", "South", "Standard", 2016, 2200, True),
    ("ST043", "Chennai", "Tamil Nadu", "South", "Standard", 2018, 1900, False),
    ("ST044", "Chennai", "Tamil Nadu", "South", "Express", 2020, 900, False),
    ("ST045", "Kochi", "Kerala", "South", "Standard", 2017, 1800, False),
    ("ST046", "Kochi", "Kerala", "South", "Express", 2021, 850, False),
    ("ST047", "Coimbatore", "Tamil Nadu", "South", "Express", 2021, 850, False),
    ("ST048", "Mysuru", "Karnataka", "South", "Express", 2022, 800, False),
    ("ST049", "Visakhapatnam", "AP", "South", "Express", 2022, 850, False),
    ("ST050", "Thiruvananthapuram", "Kerala", "South", "Express", 2023, 800, False),
    ("ST051", "Kolkata", "West Bengal", "East", "Flagship", 2015, 3600, True),
    ("ST052", "Kolkata", "West Bengal", "East", "Standard", 2016, 2100, True),
    ("ST053", "Kolkata", "West Bengal", "East", "Standard", 2018, 1800, False),
    ("ST054", "Kolkata", "West Bengal", "East", "Express", 2020, 900, False),
    ("ST055", "Bhubaneswar", "Odisha", "East", "Standard", 2019, 1600, False),
    ("ST056", "Bhubaneswar", "Odisha", "East", "Express", 2022, 850, False),
    ("ST057", "Patna", "Bihar", "East", "Express", 2022, 800, False),
    ("ST058", "Guwahati", "Assam", "East", "Standard", 2020, 1500, False),
    ("ST059", "Ranchi", "Jharkhand", "East", "Express", 2023, 800, False),
    ("ST060", "Raipur", "Chhattisgarh", "East", "Express", 2023, 800, False),
    # Online channels
    ("ON001", "Online", "National", "Online", "Online", 2013, 0, False),
    ("ON002", "Online", "National", "Online", "App", 2018, 0, False),
    # HUFT Spa standalone (not full stores)
    ("SP001", "Mumbai", "Maharashtra", "West", "Spa", 2019, 600, True),
    ("SP002", "Bengaluru", "Karnataka", "South", "Spa", 2019, 600, True),
    ("SP003", "Delhi", "Delhi", "North", "Spa", 2020, 600, True),
    ("SP004", "Hyderabad", "Telangana", "South", "Spa", 2020, 600, True),
    ("SP005", "Pune", "Maharashtra", "West", "Spa", 2021, 600, True),
]

# ── Product master — real HUFT brands + SKUs ─────────────────────────────────

PRODUCTS = [
    # ── DOG FOOD — Premium imported (Royal Canin, Farmina, Orijen) ────────────
    {
        "sku_id": "FOOD_D001",
        "name": "Royal Canin Labrador Retriever Adult 12kg",
        "brand": "Royal Canin",
        "brand_type": "Third Party",
        "category": "Dog Food",
        "subcategory": "Breed Specific Dry",
        "pet_type": "Dog",
        "life_stage": "Adult",
        "breed_suitability": "Labrador Retriever",
        "weight_kg": 12.0,
        "price_inr": 4599,
        "cost_inr": 2990,
        "supplier": "Royal Canin India",
        "lead_time_days": 10,
        "base_demand": 28,
        "is_cold_chain": False,
        "margin_pct": 35,
        "min_age_months": 15,
        "max_age_months": None,
    },
    {
        "sku_id": "FOOD_D002",
        "name": "Royal Canin Golden Retriever Adult 12kg",
        "brand": "Royal Canin",
        "brand_type": "Third Party",
        "category": "Dog Food",
        "subcategory": "Breed Specific Dry",
        "pet_type": "Dog",
        "life_stage": "Adult",
        "breed_suitability": "Golden Retriever",
        "weight_kg": 12.0,
        "price_inr": 4599,
        "cost_inr": 2990,
        "supplier": "Royal Canin India",
        "lead_time_days": 10,
        "base_demand": 25,
        "is_cold_chain": False,
        "margin_pct": 35,
        "min_age_months": 15,
        "max_age_months": None,
    },
    {
        "sku_id": "FOOD_D003",
        "name": "Royal Canin Maxi Puppy 4kg",
        "brand": "Royal Canin",
        "brand_type": "Third Party",
        "category": "Dog Food",
        "subcategory": "Breed Specific Dry",
        "pet_type": "Dog",
        "life_stage": "Puppy",
        "breed_suitability": "Large Breeds",
        "weight_kg": 4.0,
        "price_inr": 2299,
        "cost_inr": 1450,
        "supplier": "Royal Canin India",
        "lead_time_days": 10,
        "base_demand": 40,
        "is_cold_chain": False,
        "margin_pct": 37,
        "min_age_months": 2,
        "max_age_months": 15,
    },
    {
        "sku_id": "FOOD_D004",
        "name": "Royal Canin Labrador Retriever Puppy 3kg",
        "brand": "Royal Canin",
        "brand_type": "Third Party",
        "category": "Dog Food",
        "subcategory": "Breed Specific Dry",
        "pet_type": "Dog",
        "life_stage": "Puppy",
        "breed_suitability": "Labrador Retriever",
        "weight_kg": 3.0,
        "price_inr": 2099,
        "cost_inr": 1350,
        "supplier": "Royal Canin India",
        "lead_time_days": 10,
        "base_demand": 35,
        "is_cold_chain": False,
        "margin_pct": 36,
        "min_age_months": 2,
        "max_age_months": 15,
    },
    {
        "sku_id": "FOOD_D005",
        "name": "Farmina N&D Grain Free Chicken Puppy Medium 3kg",
        "brand": "Farmina",
        "brand_type": "Third Party",
        "category": "Dog Food",
        "subcategory": "Grain Free Dry",
        "pet_type": "Dog",
        "life_stage": "Puppy",
        "breed_suitability": "Medium Breeds",
        "weight_kg": 3.0,
        "price_inr": 2799,
        "cost_inr": 1750,
        "supplier": "Farmina India",
        "lead_time_days": 14,
        "base_demand": 22,
        "is_cold_chain": False,
        "margin_pct": 38,
        "min_age_months": 2,
        "max_age_months": 12,
    },
    {
        "sku_id": "FOOD_D006",
        "name": "Farmina N&D Grain Free Chicken Adult Medium 3kg",
        "brand": "Farmina",
        "brand_type": "Third Party",
        "category": "Dog Food",
        "subcategory": "Grain Free Dry",
        "pet_type": "Dog",
        "life_stage": "Adult",
        "breed_suitability": "Medium Breeds",
        "weight_kg": 3.0,
        "price_inr": 2799,
        "cost_inr": 1750,
        "supplier": "Farmina India",
        "lead_time_days": 14,
        "base_demand": 20,
        "is_cold_chain": False,
        "margin_pct": 38,
        "min_age_months": 12,
        "max_age_months": None,
    },
    {
        "sku_id": "FOOD_D007",
        "name": "Drools Focus Puppy Super Premium 3kg",
        "brand": "Drools",
        "brand_type": "Third Party",
        "category": "Dog Food",
        "subcategory": "Super Premium Dry",
        "pet_type": "Dog",
        "life_stage": "Puppy",
        "breed_suitability": "All Breeds",
        "weight_kg": 3.0,
        "price_inr": 899,
        "cost_inr": 510,
        "supplier": "Drools Pet Food",
        "lead_time_days": 7,
        "base_demand": 65,
        "is_cold_chain": False,
        "margin_pct": 43,
        "min_age_months": 1,
        "max_age_months": 12,
    },
    {
        "sku_id": "FOOD_D008",
        "name": "Drools Focus Adult Super Premium 3kg",
        "brand": "Drools",
        "brand_type": "Third Party",
        "category": "Dog Food",
        "subcategory": "Super Premium Dry",
        "pet_type": "Dog",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 3.0,
        "price_inr": 849,
        "cost_inr": 480,
        "supplier": "Drools Pet Food",
        "lead_time_days": 7,
        "base_demand": 70,
        "is_cold_chain": False,
        "margin_pct": 43,
        "min_age_months": 12,
        "max_age_months": None,
    },
    {
        "sku_id": "FOOD_D009",
        "name": "Pedigree Puppy Chicken & Milk 3kg",
        "brand": "Pedigree",
        "brand_type": "Third Party",
        "category": "Dog Food",
        "subcategory": "Standard Dry",
        "pet_type": "Dog",
        "life_stage": "Puppy",
        "breed_suitability": "All Breeds",
        "weight_kg": 3.0,
        "price_inr": 549,
        "cost_inr": 290,
        "supplier": "Mars Petcare India",
        "lead_time_days": 5,
        "base_demand": 120,
        "is_cold_chain": False,
        "margin_pct": 47,
        "min_age_months": 1,
        "max_age_months": 12,
    },
    {
        "sku_id": "FOOD_D010",
        "name": "Pedigree Adult Chicken & Vegetables 3kg",
        "brand": "Pedigree",
        "brand_type": "Third Party",
        "category": "Dog Food",
        "subcategory": "Standard Dry",
        "pet_type": "Dog",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 3.0,
        "price_inr": 499,
        "cost_inr": 260,
        "supplier": "Mars Petcare India",
        "lead_time_days": 5,
        "base_demand": 130,
        "is_cold_chain": False,
        "margin_pct": 48,
        "min_age_months": 12,
        "max_age_months": None,
    },
    # ── DOG FOOD — HUFT Private Label (Sara's Fresh, Hearty) ──────────────────
    {
        "sku_id": "FOOD_D011",
        "name": "Sara's Wholesome Chicken & Rice Fresh 1kg",
        "brand": "Sara's",
        "brand_type": "Private Label",
        "category": "Dog Food",
        "subcategory": "Fresh/Wet Daily Meals",
        "pet_type": "Dog",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 1.0,
        "price_inr": 299,
        "cost_inr": 145,
        "supplier": "Sara's Kitchen (HUFT)",
        "lead_time_days": 2,
        "base_demand": 85,
        "is_cold_chain": True,
        "margin_pct": 52,
        "min_age_months": 4,
        "max_age_months": None,
    },
    {
        "sku_id": "FOOD_D012",
        "name": "Sara's Wholesome Mutton & Vegetables Fresh 1kg",
        "brand": "Sara's",
        "brand_type": "Private Label",
        "category": "Dog Food",
        "subcategory": "Fresh/Wet Daily Meals",
        "pet_type": "Dog",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 1.0,
        "price_inr": 329,
        "cost_inr": 158,
        "supplier": "Sara's Kitchen (HUFT)",
        "lead_time_days": 2,
        "base_demand": 72,
        "is_cold_chain": True,
        "margin_pct": 52,
        "min_age_months": 4,
        "max_age_months": None,
    },
    {
        "sku_id": "FOOD_D013",
        "name": "Sara's Wholesome Puppy Chicken & Egg Fresh 500g",
        "brand": "Sara's",
        "brand_type": "Private Label",
        "category": "Dog Food",
        "subcategory": "Fresh/Wet Daily Meals",
        "pet_type": "Dog",
        "life_stage": "Puppy",
        "breed_suitability": "All Breeds",
        "weight_kg": 0.5,
        "price_inr": 199,
        "cost_inr": 95,
        "supplier": "Sara's Kitchen (HUFT)",
        "lead_time_days": 2,
        "base_demand": 60,
        "is_cold_chain": True,
        "margin_pct": 52,
        "min_age_months": 1,
        "max_age_months": 12,
    },
    {
        "sku_id": "FOOD_D014",
        "name": "Hearty Chicken Adult Dry Dog Food 3kg",
        "brand": "Hearty",
        "brand_type": "Private Label",
        "category": "Dog Food",
        "subcategory": "Standard Dry",
        "pet_type": "Dog",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 3.0,
        "price_inr": 649,
        "cost_inr": 295,
        "supplier": "Contract Manufacturer",
        "lead_time_days": 7,
        "base_demand": 90,
        "is_cold_chain": False,
        "margin_pct": 55,
        "min_age_months": 12,
        "max_age_months": None,
    },
    {
        "sku_id": "FOOD_D015",
        "name": "Hearty Puppy Starter Dry 1.5kg",
        "brand": "Hearty",
        "brand_type": "Private Label",
        "category": "Dog Food",
        "subcategory": "Standard Dry",
        "pet_type": "Dog",
        "life_stage": "Puppy",
        "breed_suitability": "All Breeds",
        "weight_kg": 1.5,
        "price_inr": 399,
        "cost_inr": 175,
        "supplier": "Contract Manufacturer",
        "lead_time_days": 7,
        "base_demand": 75,
        "is_cold_chain": False,
        "margin_pct": 56,
        "min_age_months": 1,
        "max_age_months": 12,
    },
    # ── CAT FOOD — Real brands + HUFT private label (Meowsi) ──────────────────
    {
        "sku_id": "FOOD_C001",
        "name": "Royal Canin Indoor Cat Adult 2kg",
        "brand": "Royal Canin",
        "brand_type": "Third Party",
        "category": "Cat Food",
        "subcategory": "Premium Dry",
        "pet_type": "Cat",
        "life_stage": "Adult",
        "breed_suitability": "Indoor Cats",
        "weight_kg": 2.0,
        "price_inr": 1899,
        "cost_inr": 1220,
        "supplier": "Royal Canin India",
        "lead_time_days": 10,
        "base_demand": 45,
        "is_cold_chain": False,
        "margin_pct": 36,
        "min_age_months": 12,
        "max_age_months": None,
    },
    {
        "sku_id": "FOOD_C002",
        "name": "Royal Canin Kitten 2kg",
        "brand": "Royal Canin",
        "brand_type": "Third Party",
        "category": "Cat Food",
        "subcategory": "Premium Dry",
        "pet_type": "Cat",
        "life_stage": "Kitten",
        "breed_suitability": "All Breeds",
        "weight_kg": 2.0,
        "price_inr": 1899,
        "cost_inr": 1220,
        "supplier": "Royal Canin India",
        "lead_time_days": 10,
        "base_demand": 38,
        "is_cold_chain": False,
        "margin_pct": 36,
        "min_age_months": 1,
        "max_age_months": 12,
    },
    {
        "sku_id": "FOOD_C003",
        "name": "Whiskas Adult Ocean Fish Dry 1.2kg",
        "brand": "Whiskas",
        "brand_type": "Third Party",
        "category": "Cat Food",
        "subcategory": "Standard Dry",
        "pet_type": "Cat",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 1.2,
        "price_inr": 399,
        "cost_inr": 195,
        "supplier": "Mars Petcare India",
        "lead_time_days": 5,
        "base_demand": 95,
        "is_cold_chain": False,
        "margin_pct": 51,
        "min_age_months": 12,
        "max_age_months": None,
    },
    {
        "sku_id": "FOOD_C004",
        "name": "Whiskas Kitten Milk Starter 1.2kg",
        "brand": "Whiskas",
        "brand_type": "Third Party",
        "category": "Cat Food",
        "subcategory": "Standard Dry",
        "pet_type": "Cat",
        "life_stage": "Kitten",
        "breed_suitability": "All Breeds",
        "weight_kg": 1.2,
        "price_inr": 399,
        "cost_inr": 195,
        "supplier": "Mars Petcare India",
        "lead_time_days": 5,
        "base_demand": 55,
        "is_cold_chain": False,
        "margin_pct": 51,
        "min_age_months": 1,
        "max_age_months": 12,
    },
    {
        "sku_id": "FOOD_C005",
        "name": "Meowsi Complete Meal Chicken Adult 1kg",
        "brand": "Meowsi",
        "brand_type": "Private Label",
        "category": "Cat Food",
        "subcategory": "Complete Meals",
        "pet_type": "Cat",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 1.0,
        "price_inr": 299,
        "cost_inr": 130,
        "supplier": "Contract Manufacturer",
        "lead_time_days": 7,
        "base_demand": 80,
        "is_cold_chain": False,
        "margin_pct": 57,
        "min_age_months": 12,
        "max_age_months": None,
    },
    {
        "sku_id": "FOOD_C006",
        "name": "Meowsi Kitten Complete Meal 500g",
        "brand": "Meowsi",
        "brand_type": "Private Label",
        "category": "Cat Food",
        "subcategory": "Complete Meals",
        "pet_type": "Cat",
        "life_stage": "Kitten",
        "breed_suitability": "All Breeds",
        "weight_kg": 0.5,
        "price_inr": 199,
        "cost_inr": 85,
        "supplier": "Contract Manufacturer",
        "lead_time_days": 7,
        "base_demand": 60,
        "is_cold_chain": False,
        "margin_pct": 57,
        "min_age_months": 1,
        "max_age_months": 12,
    },
    # ── DOG TREATS ────────────────────────────────────────────────────────────
    {
        "sku_id": "TRT_D001",
        "name": "Chip Chops Barbeque Chicken Strips 70g",
        "brand": "Chip Chops",
        "brand_type": "Third Party",
        "category": "Dog Treats",
        "subcategory": "Soft Chewy",
        "pet_type": "Dog",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 0.07,
        "price_inr": 149,
        "cost_inr": 72,
        "supplier": "Drools Pet Food",
        "lead_time_days": 7,
        "base_demand": 160,
        "is_cold_chain": False,
        "margin_pct": 52,
        "min_age_months": 4,
        "max_age_months": None,
    },
    {
        "sku_id": "TRT_D002",
        "name": "Gnawlers Chicken Bone Dog Treat 10-pack",
        "brand": "Gnawlers",
        "brand_type": "Third Party",
        "category": "Dog Treats",
        "subcategory": "Natural Chews",
        "pet_type": "Dog",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 0.12,
        "price_inr": 199,
        "cost_inr": 95,
        "supplier": "Gnawlers",
        "lead_time_days": 10,
        "base_demand": 120,
        "is_cold_chain": False,
        "margin_pct": 52,
        "min_age_months": 4,
        "max_age_months": None,
    },
    {
        "sku_id": "TRT_D003",
        "name": "Happi Doggy Dental Chew Vanilla 150g",
        "brand": "Happi Doggy",
        "brand_type": "Third Party",
        "category": "Dog Treats",
        "subcategory": "Dental Chews",
        "pet_type": "Dog",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 0.15,
        "price_inr": 299,
        "cost_inr": 140,
        "supplier": "Happi Doggy",
        "lead_time_days": 10,
        "base_demand": 85,
        "is_cold_chain": False,
        "margin_pct": 53,
        "min_age_months": 4,
        "max_age_months": None,
    },
    {
        "sku_id": "TRT_D004",
        "name": "Pedigree Dentastix Large 7-pack",
        "brand": "Pedigree",
        "brand_type": "Third Party",
        "category": "Dog Treats",
        "subcategory": "Dental Chews",
        "pet_type": "Dog",
        "life_stage": "Adult",
        "breed_suitability": "Large Breeds",
        "weight_kg": 0.18,
        "price_inr": 199,
        "cost_inr": 95,
        "supplier": "Mars Petcare India",
        "lead_time_days": 5,
        "base_demand": 140,
        "is_cold_chain": False,
        "margin_pct": 52,
        "min_age_months": 4,
        "max_age_months": None,
    },
    # ── CAT TREATS ────────────────────────────────────────────────────────────
    {
        "sku_id": "TRT_C001",
        "name": "Temptations Tuna Cat Treats 85g",
        "brand": "Temptations",
        "brand_type": "Third Party",
        "category": "Cat Treats",
        "subcategory": "Crunchy Treats",
        "pet_type": "Cat",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 0.085,
        "price_inr": 199,
        "cost_inr": 95,
        "supplier": "Mars Petcare India",
        "lead_time_days": 5,
        "base_demand": 110,
        "is_cold_chain": False,
        "margin_pct": 52,
        "min_age_months": 12,
        "max_age_months": None,
    },
    {
        "sku_id": "TRT_C002",
        "name": "Whiskas Temptations Salmon 40g",
        "brand": "Whiskas",
        "brand_type": "Third Party",
        "category": "Cat Treats",
        "subcategory": "Creamy Treats",
        "pet_type": "Cat",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 0.04,
        "price_inr": 99,
        "cost_inr": 45,
        "supplier": "Mars Petcare India",
        "lead_time_days": 5,
        "base_demand": 150,
        "is_cold_chain": False,
        "margin_pct": 55,
        "min_age_months": 12,
        "max_age_months": None,
    },
    {
        "sku_id": "TRT_C003",
        "name": "Kittos Creamy Cat Treat Chicken 15g x 4",
        "brand": "Kittos",
        "brand_type": "Third Party",
        "category": "Cat Treats",
        "subcategory": "Creamy Treats",
        "pet_type": "Cat",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 0.06,
        "price_inr": 149,
        "cost_inr": 68,
        "supplier": "Kittos",
        "lead_time_days": 7,
        "base_demand": 130,
        "is_cold_chain": False,
        "margin_pct": 54,
        "min_age_months": 12,
        "max_age_months": None,
    },
    # ── HEALTH & SUPPLEMENTS ──────────────────────────────────────────────────
    {
        "sku_id": "HLTH_001",
        "name": "NexGard Chewables for Dogs 4.1-10kg 3-pack",
        "brand": "NexGard",
        "brand_type": "Third Party",
        "category": "Health",
        "subcategory": "Tick Flea Control",
        "pet_type": "Dog",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 0.05,
        "price_inr": 1299,
        "cost_inr": 780,
        "supplier": "Boehringer Ingelheim",
        "lead_time_days": 14,
        "base_demand": 55,
        "is_cold_chain": False,
        "margin_pct": 40,
        "min_age_months": 8,
        "max_age_months": None,
    },
    {
        "sku_id": "HLTH_002",
        "name": "Frontline Plus for Dogs 10-20kg 3-pack",
        "brand": "Frontline",
        "brand_type": "Third Party",
        "category": "Health",
        "subcategory": "Tick Flea Control",
        "pet_type": "Dog",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 0.03,
        "price_inr": 899,
        "cost_inr": 520,
        "supplier": "Boehringer Ingelheim",
        "lead_time_days": 14,
        "base_demand": 65,
        "is_cold_chain": False,
        "margin_pct": 42,
        "min_age_months": 8,
        "max_age_months": None,
    },
    {
        "sku_id": "HLTH_003",
        "name": "Virbac Endogard Dewormer Dogs 10kg 6-tab",
        "brand": "Virbac",
        "brand_type": "Third Party",
        "category": "Health",
        "subcategory": "Dewormers",
        "pet_type": "Dog",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 0.02,
        "price_inr": 199,
        "cost_inr": 95,
        "supplier": "Virbac India",
        "lead_time_days": 10,
        "base_demand": 90,
        "is_cold_chain": False,
        "margin_pct": 52,
        "min_age_months": 3,
        "max_age_months": None,
    },
    {
        "sku_id": "HLTH_004",
        "name": "Himalaya Himcal Calcium Supplement 200g",
        "brand": "Himalaya",
        "brand_type": "Third Party",
        "category": "Health",
        "subcategory": "Supplements",
        "pet_type": "Dog",
        "life_stage": "Puppy",
        "breed_suitability": "All Breeds",
        "weight_kg": 0.2,
        "price_inr": 249,
        "cost_inr": 110,
        "supplier": "Himalaya Pet",
        "lead_time_days": 7,
        "base_demand": 80,
        "is_cold_chain": False,
        "margin_pct": 56,
        "min_age_months": 1,
        "max_age_months": 18,
    },
    {
        "sku_id": "HLTH_005",
        "name": "NutriWag Fish Oil for Dogs 200ml",
        "brand": "NutriWag",
        "brand_type": "Third Party",
        "category": "Health",
        "subcategory": "Supplements",
        "pet_type": "Dog",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 0.2,
        "price_inr": 499,
        "cost_inr": 220,
        "supplier": "NutriWag",
        "lead_time_days": 10,
        "base_demand": 55,
        "is_cold_chain": False,
        "margin_pct": 56,
        "min_age_months": 4,
        "max_age_months": None,
    },
    {
        "sku_id": "HLTH_006",
        "name": "Frontline Spot On Cat 3-pack",
        "brand": "Frontline",
        "brand_type": "Third Party",
        "category": "Health",
        "subcategory": "Tick Flea Control",
        "pet_type": "Cat",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 0.02,
        "price_inr": 799,
        "cost_inr": 460,
        "supplier": "Boehringer Ingelheim",
        "lead_time_days": 14,
        "base_demand": 45,
        "is_cold_chain": False,
        "margin_pct": 42,
        "min_age_months": 8,
        "max_age_months": None,
    },
    {
        "sku_id": "HLTH_007",
        "name": "Bravecto Chewable 250mg Dogs 4.5-10kg",
        "brand": "Bravecto",
        "brand_type": "Third Party",
        "category": "Health",
        "subcategory": "Tick Flea Control",
        "pet_type": "Dog",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 0.03,
        "price_inr": 1999,
        "cost_inr": 1220,
        "supplier": "MSD Animal Health",
        "lead_time_days": 14,
        "base_demand": 38,
        "is_cold_chain": False,
        "margin_pct": 39,
        "min_age_months": 6,
        "max_age_months": None,
    },
    # ── GROOMING ──────────────────────────────────────────────────────────────
    {
        "sku_id": "GROM_001",
        "name": "Wahl Clipper Deluxe Dog Grooming Kit",
        "brand": "Wahl",
        "brand_type": "Third Party",
        "category": "Dog Grooming",
        "subcategory": "Clippers",
        "pet_type": "Dog",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 0.5,
        "price_inr": 2499,
        "cost_inr": 1400,
        "supplier": "Wahl India",
        "lead_time_days": 14,
        "base_demand": 18,
        "is_cold_chain": False,
        "margin_pct": 44,
        "min_age_months": None,
        "max_age_months": None,
    },
    {
        "sku_id": "GROM_002",
        "name": "HUFT Freshness Dog Shampoo 250ml",
        "brand": "Heads Up For Tails",
        "brand_type": "Private Label",
        "category": "Dog Grooming",
        "subcategory": "Shampoo Conditioner",
        "pet_type": "Dog",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 0.3,
        "price_inr": 349,
        "cost_inr": 120,
        "supplier": "HUFT Beauty Lab",
        "lead_time_days": 7,
        "base_demand": 95,
        "is_cold_chain": False,
        "margin_pct": 66,
        "min_age_months": None,
        "max_age_months": None,
    },
    {
        "sku_id": "GROM_003",
        "name": "HUFT Anti-Tick Shampoo Dog 250ml",
        "brand": "Heads Up For Tails",
        "brand_type": "Private Label",
        "category": "Dog Grooming",
        "subcategory": "Shampoo Conditioner",
        "pet_type": "Dog",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 0.3,
        "price_inr": 399,
        "cost_inr": 140,
        "supplier": "HUFT Beauty Lab",
        "lead_time_days": 7,
        "base_demand": 75,
        "is_cold_chain": False,
        "margin_pct": 65,
        "min_age_months": None,
        "max_age_months": None,
    },
    {
        "sku_id": "GROM_004",
        "name": "HUFT Cat Gentle Shampoo 200ml",
        "brand": "Heads Up For Tails",
        "brand_type": "Private Label",
        "category": "Cat Grooming",
        "subcategory": "Shampoo Conditioner",
        "pet_type": "Cat",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 0.25,
        "price_inr": 299,
        "cost_inr": 105,
        "supplier": "HUFT Beauty Lab",
        "lead_time_days": 7,
        "base_demand": 55,
        "is_cold_chain": False,
        "margin_pct": 65,
        "min_age_months": None,
        "max_age_months": None,
    },
    # ── WALK ESSENTIALS ───────────────────────────────────────────────────────
    {
        "sku_id": "WALK_001",
        "name": "Dash Dog Finn Collar Medium",
        "brand": "Dash Dog",
        "brand_type": "Private Label",
        "category": "Walk Essentials",
        "subcategory": "Collars",
        "pet_type": "Dog",
        "life_stage": "Adult",
        "breed_suitability": "Medium Breeds",
        "weight_kg": 0.08,
        "price_inr": 799,
        "cost_inr": 280,
        "supplier": "HUFT Accessories",
        "lead_time_days": 10,
        "base_demand": 40,
        "is_cold_chain": False,
        "margin_pct": 65,
        "min_age_months": None,
        "max_age_months": None,
    },
    {
        "sku_id": "WALK_002",
        "name": "Dash Dog Neon Leash 120cm",
        "brand": "Dash Dog",
        "brand_type": "Private Label",
        "category": "Walk Essentials",
        "subcategory": "Leashes",
        "pet_type": "Dog",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 0.12,
        "price_inr": 699,
        "cost_inr": 240,
        "supplier": "HUFT Accessories",
        "lead_time_days": 10,
        "base_demand": 45,
        "is_cold_chain": False,
        "margin_pct": 66,
        "min_age_months": None,
        "max_age_months": None,
    },
    {
        "sku_id": "WALK_003",
        "name": "Ruffwear Front Range Harness",
        "brand": "Ruffwear",
        "brand_type": "Third Party",
        "category": "Walk Essentials",
        "subcategory": "Harnesses",
        "pet_type": "Dog",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 0.3,
        "price_inr": 3999,
        "cost_inr": 2400,
        "supplier": "Ruffwear India",
        "lead_time_days": 21,
        "base_demand": 12,
        "is_cold_chain": False,
        "margin_pct": 40,
        "min_age_months": None,
        "max_age_months": None,
    },
    {
        "sku_id": "WALK_004",
        "name": "HUFT Personalised Name Tag Dog",
        "brand": "Heads Up For Tails",
        "brand_type": "Private Label",
        "category": "Walk Essentials",
        "subcategory": "Name Tags",
        "pet_type": "Dog",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 0.02,
        "price_inr": 299,
        "cost_inr": 80,
        "supplier": "HUFT Accessories",
        "lead_time_days": 5,
        "base_demand": 60,
        "is_cold_chain": False,
        "margin_pct": 73,
        "min_age_months": None,
        "max_age_months": None,
    },
    # ── TOYS ──────────────────────────────────────────────────────────────────
    {
        "sku_id": "TOYS_D001",
        "name": "KONG Classic Dog Toy Medium",
        "brand": "KONG",
        "brand_type": "Third Party",
        "category": "Dog Toys",
        "subcategory": "Chew Toys",
        "pet_type": "Dog",
        "life_stage": "Adult",
        "breed_suitability": "Medium Breeds",
        "weight_kg": 0.15,
        "price_inr": 899,
        "cost_inr": 480,
        "supplier": "KONG India",
        "lead_time_days": 14,
        "base_demand": 35,
        "is_cold_chain": False,
        "margin_pct": 47,
        "min_age_months": None,
        "max_age_months": None,
    },
    {
        "sku_id": "TOYS_D002",
        "name": "Trixie Dog Activity Flip Board Level 2",
        "brand": "Trixie",
        "brand_type": "Third Party",
        "category": "Dog Toys",
        "subcategory": "Interactive Puzzle",
        "pet_type": "Dog",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 0.45,
        "price_inr": 1299,
        "cost_inr": 720,
        "supplier": "Trixie India",
        "lead_time_days": 14,
        "base_demand": 22,
        "is_cold_chain": False,
        "margin_pct": 45,
        "min_age_months": None,
        "max_age_months": None,
    },
    {
        "sku_id": "TOYS_D003",
        "name": "HUFT Plush Squeaky Dog Toy Avocado",
        "brand": "Heads Up For Tails",
        "brand_type": "Private Label",
        "category": "Dog Toys",
        "subcategory": "Plush Toys",
        "pet_type": "Dog",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 0.12,
        "price_inr": 499,
        "cost_inr": 145,
        "supplier": "HUFT Toys",
        "lead_time_days": 10,
        "base_demand": 70,
        "is_cold_chain": False,
        "margin_pct": 71,
        "min_age_months": None,
        "max_age_months": None,
    },
    {
        "sku_id": "TOYS_C001",
        "name": "Trixie Cat Activity Fun Board",
        "brand": "Trixie",
        "brand_type": "Third Party",
        "category": "Cat Toys",
        "subcategory": "Interactive Toys",
        "pet_type": "Cat",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 0.35,
        "price_inr": 999,
        "cost_inr": 560,
        "supplier": "Trixie India",
        "lead_time_days": 14,
        "base_demand": 28,
        "is_cold_chain": False,
        "margin_pct": 44,
        "min_age_months": None,
        "max_age_months": None,
    },
    {
        "sku_id": "TOYS_C002",
        "name": "HUFT Catnip Crinkle Tunnel",
        "brand": "Heads Up For Tails",
        "brand_type": "Private Label",
        "category": "Cat Toys",
        "subcategory": "Tunnels",
        "pet_type": "Cat",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 0.25,
        "price_inr": 799,
        "cost_inr": 220,
        "supplier": "HUFT Toys",
        "lead_time_days": 10,
        "base_demand": 45,
        "is_cold_chain": False,
        "margin_pct": 72,
        "min_age_months": None,
        "max_age_months": None,
    },
    # ── BEDDING ───────────────────────────────────────────────────────────────
    {
        "sku_id": "BED_001",
        "name": "HUFT Luxe Plush Dog Bed Large",
        "brand": "Heads Up For Tails",
        "brand_type": "Private Label",
        "category": "Dog Bedding",
        "subcategory": "Beds",
        "pet_type": "Dog",
        "life_stage": "Adult",
        "breed_suitability": "Large Breeds",
        "weight_kg": 1.2,
        "price_inr": 2499,
        "cost_inr": 780,
        "supplier": "HUFT Home",
        "lead_time_days": 10,
        "base_demand": 20,
        "is_cold_chain": False,
        "margin_pct": 69,
        "min_age_months": None,
        "max_age_months": None,
    },
    {
        "sku_id": "BED_002",
        "name": "HUFT Cooling Mat Dog Large",
        "brand": "Heads Up For Tails",
        "brand_type": "Private Label",
        "category": "Dog Bedding",
        "subcategory": "Mats",
        "pet_type": "Dog",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 0.8,
        "price_inr": 1499,
        "cost_inr": 460,
        "supplier": "HUFT Home",
        "lead_time_days": 10,
        "base_demand": 30,
        "is_cold_chain": False,
        "margin_pct": 69,
        "min_age_months": None,
        "max_age_months": None,
    },
    {
        "sku_id": "BED_003",
        "name": "HUFT Cat Tree 120cm",
        "brand": "Heads Up For Tails",
        "brand_type": "Private Label",
        "category": "Cat Supplies",
        "subcategory": "Trees Scratchers",
        "pet_type": "Cat",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 8.0,
        "price_inr": 4999,
        "cost_inr": 1800,
        "supplier": "HUFT Home",
        "lead_time_days": 14,
        "base_demand": 10,
        "is_cold_chain": False,
        "margin_pct": 64,
        "min_age_months": None,
        "max_age_months": None,
    },
    # ── CAT LITTER ────────────────────────────────────────────────────────────
    {
        "sku_id": "LITT_001",
        "name": "Cats Best Original Clumping Litter 10L",
        "brand": "Cats Best",
        "brand_type": "Third Party",
        "category": "Cat Supplies",
        "subcategory": "Litter",
        "pet_type": "Cat",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 4.3,
        "price_inr": 999,
        "cost_inr": 580,
        "supplier": "Cats Best India",
        "lead_time_days": 14,
        "base_demand": 70,
        "is_cold_chain": False,
        "margin_pct": 42,
        "min_age_months": None,
        "max_age_months": None,
    },
    {
        "sku_id": "LITT_002",
        "name": "HUFT Clumping Sand Cat Litter 5kg",
        "brand": "Heads Up For Tails",
        "brand_type": "Private Label",
        "category": "Cat Supplies",
        "subcategory": "Litter",
        "pet_type": "Cat",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 5.0,
        "price_inr": 699,
        "cost_inr": 210,
        "supplier": "Contract Manufacturer",
        "lead_time_days": 7,
        "base_demand": 85,
        "is_cold_chain": False,
        "margin_pct": 70,
        "min_age_months": None,
        "max_age_months": None,
    },
    # ── CLOTHING ──────────────────────────────────────────────────────────────
    {
        "sku_id": "CLTH_001",
        "name": "HUFT Pawfect Hoodie Dog Medium",
        "brand": "Heads Up For Tails",
        "brand_type": "Private Label",
        "category": "Dog Clothing",
        "subcategory": "Sweatshirts",
        "pet_type": "Dog",
        "life_stage": "Adult",
        "breed_suitability": "Medium Breeds",
        "weight_kg": 0.2,
        "price_inr": 1299,
        "cost_inr": 360,
        "supplier": "HUFT Apparel",
        "lead_time_days": 14,
        "base_demand": 25,
        "is_cold_chain": False,
        "margin_pct": 72,
        "min_age_months": None,
        "max_age_months": None,
    },
    {
        "sku_id": "CLTH_002",
        "name": "HUFT Waterproof Dog Raincoat Large",
        "brand": "Heads Up For Tails",
        "brand_type": "Private Label",
        "category": "Dog Clothing",
        "subcategory": "Raincoats",
        "pet_type": "Dog",
        "life_stage": "Adult",
        "breed_suitability": "Large Breeds",
        "weight_kg": 0.3,
        "price_inr": 1499,
        "cost_inr": 420,
        "supplier": "HUFT Apparel",
        "lead_time_days": 14,
        "base_demand": 20,
        "is_cold_chain": False,
        "margin_pct": 72,
        "min_age_months": None,
        "max_age_months": None,
    },
    {
        "sku_id": "CLTH_003",
        "name": "HUFT Bandana Dog Personalised",
        "brand": "Heads Up For Tails",
        "brand_type": "Private Label",
        "category": "Dog Clothing",
        "subcategory": "Bow Ties Bandanas",
        "pet_type": "Dog",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 0.05,
        "price_inr": 399,
        "cost_inr": 90,
        "supplier": "HUFT Apparel",
        "lead_time_days": 7,
        "base_demand": 55,
        "is_cold_chain": False,
        "margin_pct": 77,
        "min_age_months": None,
        "max_age_months": None,
    },
    # ── BOWLS & FEEDING ───────────────────────────────────────────────────────
    {
        "sku_id": "BOWL_001",
        "name": "HUFT Stainless Steel Dog Bowl Set",
        "brand": "Heads Up For Tails",
        "brand_type": "Private Label",
        "category": "Dog Bowls",
        "subcategory": "Bowls",
        "pet_type": "Dog",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 0.4,
        "price_inr": 699,
        "cost_inr": 195,
        "supplier": "HUFT Home",
        "lead_time_days": 10,
        "base_demand": 40,
        "is_cold_chain": False,
        "margin_pct": 72,
        "min_age_months": None,
        "max_age_months": None,
    },
    {
        "sku_id": "BOWL_002",
        "name": "HUFT Slow Feeder Dog Bowl",
        "brand": "Heads Up For Tails",
        "brand_type": "Private Label",
        "category": "Dog Bowls",
        "subcategory": "Bowls",
        "pet_type": "Dog",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 0.3,
        "price_inr": 599,
        "cost_inr": 165,
        "supplier": "HUFT Home",
        "lead_time_days": 10,
        "base_demand": 35,
        "is_cold_chain": False,
        "margin_pct": 72,
        "min_age_months": None,
        "max_age_months": None,
    },
    {
        "sku_id": "BOWL_003",
        "name": "HUFT Cat Ceramic Bowl Set",
        "brand": "Heads Up For Tails",
        "brand_type": "Private Label",
        "category": "Cat Bowls",
        "subcategory": "Bowls",
        "pet_type": "Cat",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 0.35,
        "price_inr": 549,
        "cost_inr": 155,
        "supplier": "HUFT Home",
        "lead_time_days": 10,
        "base_demand": 30,
        "is_cold_chain": False,
        "margin_pct": 72,
        "min_age_months": None,
        "max_age_months": None,
    },
    # ── TRAVEL & CARRIERS ─────────────────────────────────────────────────────
    {
        "sku_id": "TRVL_001",
        "name": "HUFT Explorer Pet Carrier Bag Medium",
        "brand": "Heads Up For Tails",
        "brand_type": "Private Label",
        "category": "Travel Supplies",
        "subcategory": "Carriers",
        "pet_type": "Dog",
        "life_stage": "Adult",
        "breed_suitability": "Small Breeds",
        "weight_kg": 0.7,
        "price_inr": 2499,
        "cost_inr": 720,
        "supplier": "HUFT Accessories",
        "lead_time_days": 10,
        "base_demand": 18,
        "is_cold_chain": False,
        "margin_pct": 71,
        "min_age_months": None,
        "max_age_months": None,
    },
    {
        "sku_id": "TRVL_002",
        "name": "HUFT Car Seat Belt Dog",
        "brand": "Heads Up For Tails",
        "brand_type": "Private Label",
        "category": "Travel Supplies",
        "subcategory": "Travel Aids",
        "pet_type": "Dog",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 0.1,
        "price_inr": 599,
        "cost_inr": 165,
        "supplier": "HUFT Accessories",
        "lead_time_days": 10,
        "base_demand": 32,
        "is_cold_chain": False,
        "margin_pct": 72,
        "min_age_months": None,
        "max_age_months": None,
    },
    # ── SPA CONSUMABLES (sold at HUFT Spa locations) ──────────────────────────
    {
        "sku_id": "SPA_001",
        "name": "HUFT Spa De-Shed Shampoo Professional 1L",
        "brand": "Heads Up For Tails",
        "brand_type": "Private Label",
        "category": "Grooming Consumables",
        "subcategory": "Professional Shampoo",
        "pet_type": "Dog",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 1.0,
        "price_inr": 899,
        "cost_inr": 220,
        "supplier": "HUFT Beauty Lab",
        "lead_time_days": 7,
        "base_demand": 22,
        "is_cold_chain": False,
        "margin_pct": 75,
        "min_age_months": None,
        "max_age_months": None,
    },
    {
        "sku_id": "SPA_002",
        "name": "HUFT Spa Conditioner Professional 1L",
        "brand": "Heads Up For Tails",
        "brand_type": "Private Label",
        "category": "Grooming Consumables",
        "subcategory": "Professional Conditioner",
        "pet_type": "Dog",
        "life_stage": "Adult",
        "breed_suitability": "All Breeds",
        "weight_kg": 1.0,
        "price_inr": 799,
        "cost_inr": 195,
        "supplier": "HUFT Beauty Lab",
        "lead_time_days": 7,
        "base_demand": 20,
        "is_cold_chain": False,
        "margin_pct": 76,
        "min_age_months": None,
        "max_age_months": None,
    },
    # ── GIFT CARDS ────────────────────────────────────────────────────────────
    {
        "sku_id": "GIFT_001",
        "name": "HUFT Gift Card INR 500",
        "brand": "Heads Up For Tails",
        "brand_type": "Private Label",
        "category": "Gift Cards",
        "subcategory": "Gift Cards",
        "pet_type": "All",
        "life_stage": "All",
        "breed_suitability": "All Breeds",
        "weight_kg": 0.01,
        "price_inr": 500,
        "cost_inr": 500,
        "supplier": "HUFT",
        "lead_time_days": 1,
        "base_demand": 15,
        "is_cold_chain": False,
        "margin_pct": 0,
        "min_age_months": None,
        "max_age_months": None,
    },
    {
        "sku_id": "GIFT_002",
        "name": "HUFT Gift Card INR 1000",
        "brand": "Heads Up For Tails",
        "brand_type": "Private Label",
        "category": "Gift Cards",
        "subcategory": "Gift Cards",
        "pet_type": "All",
        "life_stage": "All",
        "breed_suitability": "All Breeds",
        "weight_kg": 0.01,
        "price_inr": 1000,
        "cost_inr": 1000,
        "supplier": "HUFT",
        "lead_time_days": 1,
        "base_demand": 10,
        "is_cold_chain": False,
        "margin_pct": 0,
        "min_age_months": None,
        "max_age_months": None,
    },
]

# ── Promotions master ─────────────────────────────────────────────────────────

PROMOTIONS = [
    # format: promo_id, name, start, end, discount_pct, channel, target_category, budget_inr
    (
        "PROMO_001",
        "Diwali Mega Sale 2023",
        "2023-11-10",
        "2023-11-14",
        25,
        "All",
        "All",
        500000,
    ),
    (
        "PROMO_002",
        "New Year Sale 2024",
        "2024-01-01",
        "2024-01-05",
        20,
        "Online",
        "All",
        200000,
    ),
    (
        "PROMO_003",
        "Republic Day Dog Food Offer",
        "2024-01-24",
        "2024-01-26",
        15,
        "Online",
        "Dog Food",
        80000,
    ),
    (
        "PROMO_004",
        "Valentine's Day Pet Gifts",
        "2024-02-10",
        "2024-02-14",
        10,
        "All",
        "Accessories",
        60000,
    ),
    (
        "PROMO_005",
        "Holi Pet Care Sale",
        "2024-03-24",
        "2024-03-26",
        20,
        "Online",
        "Dog Grooming",
        70000,
    ),
    (
        "PROMO_006",
        "Monsoon Tick Flea Drive",
        "2024-06-01",
        "2024-06-30",
        15,
        "All",
        "Health",
        120000,
    ),
    (
        "PROMO_007",
        "Independence Day Sale",
        "2024-08-13",
        "2024-08-15",
        20,
        "All",
        "All",
        300000,
    ),
    (
        "PROMO_008",
        "Onam Special Kerala Stores",
        "2024-09-05",
        "2024-09-08",
        15,
        "Offline",
        "Dog Food",
        50000,
    ),
    (
        "PROMO_009",
        "Navratri Pet Food Offer",
        "2024-10-03",
        "2024-10-12",
        10,
        "Online",
        "Cat Food",
        60000,
    ),
    (
        "PROMO_010",
        "Diwali Mega Sale 2024",
        "2024-10-28",
        "2024-11-01",
        30,
        "All",
        "All",
        600000,
    ),
    (
        "PROMO_011",
        "Pet Parents Month Nov",
        "2024-11-01",
        "2024-11-30",
        12,
        "Online",
        "All",
        400000,
    ),
    (
        "PROMO_012",
        "Christmas & New Year 2024",
        "2024-12-22",
        "2024-12-31",
        20,
        "All",
        "Toys",
        200000,
    ),
    (
        # BUG-033 fix: renamed from "Sara's Fresh Tuesday" to avoid implying weekly cadence.
        # A 2-year continuous promo contaminated all Dog Food ML promo features.
        # Renamed to "Fresh Food Online Discount" — represents a permanent online pricing strategy.
        "PROMO_013",
        "Fresh Food Online Discount",
        "2023-01-01",
        "2024-12-31",
        10,
        "Online",
        "Dog Food",
        50000,
    ),
    (
        "PROMO_014",
        "App-Only Flash Sale",
        "2024-03-15",
        "2024-03-15",
        30,
        "App",
        "All",
        100000,
    ),
    (
        "PROMO_015",
        "Franchise Store Grand Opening Bengaluru",
        "2024-05-01",
        "2024-05-03",
        25,
        "Offline",
        "All",
        75000,
    ),
    (
        "PROMO_016",
        "Royal Canin Breed Week",
        "2024-04-07",
        "2024-04-13",
        10,
        "All",
        "Dog Food",
        90000,
    ),
    (
        "PROMO_017",
        "Summer Cool Pets Campaign",
        "2024-04-15",
        "2024-05-31",
        15,
        "Online",
        "Dog Bedding",
        80000,
    ),
    (
        "PROMO_018",
        "Monsoon Must-Haves",
        "2024-07-01",
        "2024-07-31",
        20,
        "All",
        "Health",
        150000,
    ),
    (
        "PROMO_019",
        "Pedigree Pairing Deal",
        "2024-08-01",
        "2024-08-31",
        12,
        "Offline",
        "Dog Food",
        70000,
    ),
    (
        "PROMO_020",
        "HUFT Foundation Donation Drive",
        "2024-09-20",
        "2024-09-30",
        0,
        "All",
        "All",
        0,
    ),
    (
        "PROMO_021",
        "Cat Parent Appreciation Day",
        "2024-08-08",
        "2024-08-08",
        20,
        "All",
        "Cat Food",
        40000,
    ),
    (
        "PROMO_022",
        "Dussehra Deals",
        "2024-10-12",
        "2024-10-13",
        15,
        "All",
        "All",
        150000,
    ),
    (
        "PROMO_023",
        "Buy 2 Get 1 Treats",
        "2024-11-15",
        "2024-11-22",
        33,
        "Offline",
        "Dog Treats",
        60000,
    ),
    (
        "PROMO_024",
        "Year End Clearance",
        "2023-12-26",
        "2023-12-31",
        40,
        "All",
        "All",
        200000,
    ),
]

# ── Suppliers updated for HUFT ─────────────────────────────────────────────────

SUPPLIERS_HUFT = [
    (
        "Royal Canin India",
        "india@royalcanin.com",
        "India",
        96.5,
        4.8,
        50,
        True,
        "Premium breed-specific food. Very reliable. Monthly consolidated orders.",
    ),
    (
        "Mars Petcare India",
        "supply@marspetcare.in",
        "India",
        97.2,
        4.7,
        100,
        True,
        "Pedigree, Whiskas, Temptations, Dentastix. Largest FMCG pet supplier in India.",
    ),
    (
        "Farmina India",
        "india@farmina.com",
        "Italy",
        92.0,
        4.9,
        30,
        False,
        "Premium grain-free range. European import. 14-day lead time.",
    ),
    (
        "Drools Pet Food",
        "orders@drools.in",
        "India",
        95.5,
        4.5,
        50,
        True,
        "Domestic super-premium. Fast turnaround from Hyderabad warehouse.",
    ),
    (
        "Boehringer Ingelheim",
        "vet@boehringer.in",
        "Germany",
        94.0,
        5.0,
        20,
        False,
        "NexGard, Frontline, Bravecto. Regulated pharma. 14-day cold-chain import.",
    ),
    (
        "MSD Animal Health",
        "orders@msd-animal.in",
        "Netherlands",
        93.5,
        4.9,
        20,
        False,
        "Bravecto. Pharma import. Strictly regulated lead times.",
    ),
    (
        "Virbac India",
        "orders@virbac.in",
        "France",
        91.5,
        4.8,
        30,
        False,
        "Dewormers, ear drops. EU import. Consistent quality.",
    ),
    (
        "Himalaya Pet",
        "orders@himalaya.co.in",
        "India",
        96.0,
        4.4,
        100,
        True,
        "Domestic Ayurvedic supplements. Budget-friendly, high volume.",
    ),
    (
        "KONG India",
        "india@kongcompany.com",
        "United States",
        90.0,
        4.7,
        20,
        False,
        "KONG toys. US import. 14-21 day lead time via distributor.",
    ),
    (
        "Trixie India",
        "india@trixie.de",
        "Germany",
        89.5,
        4.6,
        30,
        False,
        "European pet accessories and toys. 14-day import.",
    ),
    (
        "Ruffwear India",
        "india@ruffwear.com",
        "United States",
        88.0,
        4.8,
        10,
        False,
        "Premium outdoor harnesses. Small batches. 21-day import.",
    ),
    (
        "Wahl India",
        "india@wahl.com",
        "United States",
        91.0,
        4.5,
        20,
        False,
        "Professional grooming clippers. 14-day import.",
    ),
    (
        "Cats Best India",
        "india@catsbest.de",
        "Germany",
        90.5,
        4.7,
        20,
        False,
        "Premium cat litter. European import. 14-day lead time.",
    ),
    (
        "Chip Chops / Drools",
        "orders@drools.in",
        "India",
        95.5,
        4.5,
        100,
        True,
        "Chip Chops range under Drools umbrella. Domestic.",
    ),
    (
        "Gnawlers",
        "orders@gnawlers.in",
        "India",
        94.0,
        4.3,
        100,
        True,
        "Natural chews. Domestic manufacturer.",
    ),
    (
        "Happi Doggy",
        "orders@happidoggy.com",
        "India",
        93.0,
        4.4,
        100,
        True,
        "Dental chews. Domestic. Good fill rates.",
    ),
    (
        "Kittos",
        "orders@kittos.in",
        "India",
        94.5,
        4.4,
        100,
        True,
        "Cat treats. Domestic. Fast delivery.",
    ),
    (
        "NutriWag",
        "orders@nutriwag.in",
        "India",
        93.0,
        4.3,
        50,
        True,
        "Supplements. Domestic. Growing brand.",
    ),
    (
        "Sara's Kitchen (HUFT)",
        "sara@huft.in",
        "India",
        99.0,
        5.0,
        1,
        True,
        "HUFT in-house fresh food. Daily production. Cold chain mandatory.",
    ),
    (
        "Contract Manufacturer",
        "cm@huft.in",
        "India",
        97.0,
        4.6,
        50,
        True,
        "HUFT private label dry food and litter. Dedicated facility.",
    ),
    (
        "HUFT Beauty Lab",
        "beauty@huft.in",
        "India",
        98.0,
        4.7,
        20,
        True,
        "HUFT in-house grooming products. Weekly batches.",
    ),
    (
        "HUFT Accessories",
        "accessories@huft.in",
        "India",
        97.5,
        4.6,
        10,
        True,
        "Dash Dog, HUFT branded accessories. In-house design, contract manufacture.",
    ),
    (
        "HUFT Home",
        "home@huft.in",
        "India",
        97.0,
        4.5,
        5,
        True,
        "Beds, bowls, cat trees. In-house design, contract manufacture.",
    ),
    (
        "HUFT Toys",
        "toys@huft.in",
        "India",
        97.0,
        4.5,
        10,
        True,
        "HUFT branded plush and cat toys. Contract manufacture.",
    ),
    (
        "HUFT Apparel",
        "apparel@huft.in",
        "India",
        96.5,
        4.6,
        10,
        True,
        "Dog clothing. HUFT in-house design. Seasonal collections.",
    ),
    ("HUFT", "huft@huft.in", "India", 100.0, 5.0, 1, True, "Gift cards. Instant."),
]

# ── India-specific seasonality calendar ───────────────────────────────────────


def indian_seasonality(
    sku: dict, day_of_year: np.ndarray, year: np.ndarray
) -> np.ndarray:
    """Return a multiplier array shaped like day_of_year."""
    s = np.ones(len(day_of_year))

    # Diwali (approx Oct 20 – Nov 5, day 293-310)
    diwali_peak = np.exp(-0.5 * ((day_of_year - 300) / 10) ** 2)
    s += 0.45 * diwali_peak

    # Christmas / New Year (day 355-365)
    nye = np.exp(-0.5 * ((day_of_year - 360) / 8) ** 2)
    s += 0.25 * nye

    # Summer (April-May, day 91-150) — cooling mats, travel, grooming
    if sku["category"] in (
        "Dog Bedding",
        "Dog Grooming",
        "Cat Grooming",
        "Travel Supplies",
    ):
        summer = np.exp(-0.5 * ((day_of_year - 120) / 25) ** 2)
        s += 0.35 * summer

    # Monsoon (June-Sept, day 152-274) — tick & flea, anti-tick shampoo
    if sku["subcategory"] in ("Tick Flea Control",) or "Anti-Tick" in sku["name"]:
        monsoon = np.where((day_of_year >= 152) & (day_of_year <= 274), 0.5, 0.0)
        s += monsoon

    # Winter (Nov-Jan, day 305-365 + 1-30) — dog clothing, beds
    if sku["category"] in ("Dog Clothing", "Dog Bedding"):
        winter_a = np.exp(-0.5 * ((day_of_year - 340) / 20) ** 2)
        winter_b = np.exp(-0.5 * ((day_of_year - 15) / 20) ** 2)
        s += 0.40 * (winter_a + winter_b)

    # Long-term trend: HUFT is growing ~35% YoY
    trend = 1.0 + 0.35 * ((year - 2023) + (day_of_year / 365))
    s *= np.maximum(trend, 0.85)

    return s


def make_demand_huft(sku: dict, dates: pd.DatetimeIndex) -> np.ndarray:
    doy = dates.dayofyear.to_numpy()
    yr = dates.year.to_numpy()
    base = sku["base_demand"]
    seas = indian_seasonality(sku, doy, yr)
    noise = rng.normal(1.0, 0.12, len(dates))
    demand = base * seas * noise

    # Random promotional spikes (Diwali, Republic Day, etc.)
    n_spikes = max(1, len(dates) // 45)
    spike_idx = rng.integers(0, len(dates), n_spikes)
    demand[spike_idx] *= rng.uniform(1.8, 3.5, n_spikes)

    # Cold-chain SKUs have higher daily demand variance
    if sku["is_cold_chain"]:
        demand *= rng.normal(1.0, 0.20, len(dates))

    return np.maximum(np.round(demand).astype(int), 0)


def make_inventory_huft(
    demand: np.ndarray, lead_time: int, is_cold: bool
) -> np.ndarray:
    # Cold-chain SKUs have lower max inventory (shelf life constraint)
    multiplier = 5 if not is_cold else 1.5
    # BUG-23 fix: use full-period average demand, not just the first 30 days.
    # Using only January demand (the lowest period) understocks SKUs during
    # Diwali/Holi peaks, creating artificial festival stockouts in training data.
    avg_demand = float(demand.mean()) if len(demand) > 0 else 1.0
    reorder_point = 2.0 * lead_time * avg_demand
    reorder_qty = multiplier * lead_time * avg_demand
    inv = np.zeros(len(demand), dtype=int)
    inv[0] = int(reorder_qty * rng.uniform(1.1, 1.8))
    for t in range(1, len(demand)):
        v = inv[t - 1] - demand[t]
        if v <= reorder_point:
            v += int(reorder_qty * rng.uniform(0.85, 1.15))
        inv[t] = max(v, 0)
    return inv


# ── Customer segments ─────────────────────────────────────────────────────────

CUSTOMER_SEGMENTS = [
    "New Pet Parent",
    "Loyal Premium",
    "Budget Conscious",
    "Cat Specialist",
    "Multi-Pet Household",
    "Health Focused",
    "Gift Buyer",
]

CITIES = [
    "Mumbai",
    "Delhi",
    "Bengaluru",
    "Hyderabad",
    "Chennai",
    "Pune",
    "Kolkata",
    "Ahmedabad",
    "Chandigarh",
    "Kochi",
    "Jaipur",
    "Lucknow",
    "Guwahati",
    "Bhubaneswar",
]

# ── Main generator ─────────────────────────────────────────────────────────────


def generate():
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")  # 730 days
    n_days = len(dates)

    # ── 1. huft_products.csv ──────────────────────────────────────────────────
    prod_df = pd.DataFrame(PRODUCTS)
    prod_df.to_csv(OUT / "huft_products.csv", index=False)
    print(f"huft_products.csv  — {len(prod_df)} products")

    # ── 2. huft_stores.csv ───────────────────────────────────────────────────
    store_cols = [
        "store_id",
        "city",
        "state",
        "region",
        "store_type",
        "opened_year",
        "size_sqft",
        "has_spa",
    ]
    store_df = pd.DataFrame(STORES, columns=store_cols)
    store_df.to_csv(OUT / "huft_stores.csv", index=False)
    print(f"huft_stores.csv    — {len(store_df)} stores")

    # ── 3. huft_daily_demand.csv ─────────────────────────────────────────────
    demand_rows = []
    for sku in PRODUCTS:
        demand = make_demand_huft(sku, dates)
        inv = make_inventory_huft(demand, sku["lead_time_days"], sku["is_cold_chain"])
        for i, d in enumerate(dates):
            demand_rows.append(
                {
                    "date": d.strftime("%Y-%m-%d"),
                    "sku_id": sku["sku_id"],
                    "name": sku["name"],
                    "brand": sku["brand"],
                    "brand_type": sku["brand_type"],
                    "category": sku["category"],
                    "subcategory": sku["subcategory"],
                    "pet_type": sku["pet_type"],
                    "life_stage": sku["life_stage"],
                    "supplier": sku["supplier"],
                    "demand": int(demand[i]),
                    "inventory": int(inv[i]),
                    "lead_time_days": sku["lead_time_days"],
                    "price_inr": sku["price_inr"],
                    "cost_inr": sku["cost_inr"],
                    "margin_pct": sku["margin_pct"],
                    "is_cold_chain": sku["is_cold_chain"],
                }
            )
    demand_df = pd.DataFrame(demand_rows)
    demand_df.to_csv(OUT / "huft_daily_demand.csv", index=False)
    print(
        f"huft_daily_demand.csv — {len(demand_df):,} rows ({len(PRODUCTS)} SKUs × {n_days} days)"
    )

    # ── 4. huft_customers.csv ────────────────────────────────────────────────
    n_cust = 5000
    cust_rows = []
    for i in range(n_cust):
        join = pd.Timestamp(
            rng.choice(pd.date_range("2020-01-01", "2024-11-30", freq="D"))
        )
        seg = rng.choice(
            CUSTOMER_SEGMENTS, p=[0.20, 0.18, 0.22, 0.12, 0.10, 0.08, 0.10]
        )
        city = rng.choice(CITIES)
        ltv = {
            "New Pet Parent": 2500,
            "Loyal Premium": 18000,
            "Budget Conscious": 4500,
            "Cat Specialist": 8000,
            "Multi-Pet Household": 22000,
            "Health Focused": 12000,
            "Gift Buyer": 3000,
        }[seg]
        ltv = float(ltv * rng.uniform(0.5, 2.0))
        cust_rows.append(
            {
                "customer_id": f"CUST_{i + 1:05d}",
                "city": city,
                "segment": seg,
                "joined_date": join.strftime("%Y-%m-%d"),
                "pet_type": rng.choice(["Dog", "Cat", "Both"], p=[0.55, 0.30, 0.15]),
                "total_orders": int(rng.integers(1, 48)),
                "lifetime_value_inr": round(ltv, 2),
                "channel_preference": rng.choice(
                    ["Online", "Offline", "Both"], p=[0.45, 0.30, 0.25]
                ),
                "is_spa_customer": bool(rng.random() < 0.18),
                "breed": rng.choice(
                    [
                        "Labrador Retriever",
                        "Golden Retriever",
                        "German Shepherd",
                        "Beagle",
                        "Indie/Mix",
                        "Persian Cat",
                        "British Shorthair",
                        "Shih Tzu",
                        "Pomeranian",
                        "None",
                    ],
                    p=[0.14, 0.10, 0.08, 0.07, 0.20, 0.08, 0.05, 0.06, 0.05, 0.17],
                ),
            }
        )
    cust_df = pd.DataFrame(cust_rows)
    cust_df.to_csv(OUT / "huft_customers.csv", index=False)
    print(f"huft_customers.csv — {len(cust_df):,} customers")

    # ── 5. huft_promotions.csv ───────────────────────────────────────────────
    promo_cols = [
        "promo_id",
        "name",
        "start_date",
        "end_date",
        "discount_pct",
        "channel",
        "target_category",
        "budget_inr",
    ]
    promo_df = pd.DataFrame(PROMOTIONS, columns=promo_cols)
    promo_df["start_date"] = pd.to_datetime(promo_df["start_date"])
    promo_df["end_date"] = pd.to_datetime(promo_df["end_date"])
    promo_df["duration_days"] = (
        promo_df["end_date"] - promo_df["start_date"]
    ).dt.days + 1
    # Simulate revenue_generated_inr and units_sold
    promo_df["revenue_generated_inr"] = (
        (promo_df["budget_inr"] * rng.uniform(2.5, 8.0, len(promo_df)))
        .round(0)
        .astype(int)
    )
    promo_df["units_sold"] = (
        (promo_df["revenue_generated_inr"] / 650).round(0).astype(int)
    )
    promo_df.to_csv(OUT / "huft_promotions.csv", index=False)
    print(f"huft_promotions.csv — {len(promo_df)} promotions")

    # ── 6. huft_sales_transactions.csv ───────────────────────────────────────
    n_txn = 50000
    skus_list = [p["sku_id"] for p in PRODUCTS]
    sku_prices = {p["sku_id"]: p["price_inr"] for p in PRODUCTS}
    sku_cats = {p["sku_id"]: p["category"] for p in PRODUCTS}
    sku_brands = {p["sku_id"]: p["brand"] for p in PRODUCTS}
    sku_costs = {p["sku_id"]: p["cost_inr"] for p in PRODUCTS}
    # Demand-weighted SKU sampling
    weights = np.array([p["base_demand"] for p in PRODUCTS], dtype=float)
    weights /= weights.sum()
    txn_rows = []
    for i in range(n_txn):
        txn_date = pd.Timestamp(
            rng.choice(pd.date_range("2023-01-01", "2024-12-31", freq="D"))
        )
        sku = rng.choice(skus_list, p=weights)
        price = sku_prices[sku]
        qty = int(rng.integers(1, 4))
        channel = rng.choice(["Online", "Offline", "App"], p=[0.48, 0.42, 0.10])
        city = rng.choice(CITIES)
        seg = rng.choice(CUSTOMER_SEGMENTS)
        # Apply promo discount if date falls in a promo window
        discount = 0.0
        for row in PROMOTIONS:
            pstart = pd.Timestamp(row[2])
            pend = pd.Timestamp(row[3])
            if pstart <= txn_date <= pend:
                # BUG-025 fix: explicit parentheses to prevent reader confusion
                # about operator precedence (and binds tighter than or)
                if (row[5] in ("All", channel)) or (
                    row[5] == "Online" and channel in ("Online", "App")
                ):
                    if row[6] == "All" or row[6] == sku_cats[sku]:
                        discount = row[4] / 100.0
                        break
        net_price = round(price * (1 - discount) * qty, 2)
        gross_margin = round(
            (sku_prices[sku] - sku_costs[sku]) * qty * (1 - discount), 2
        )
        txn_rows.append(
            {
                "txn_id": f"TXN_{i + 1:06d}",
                "date": txn_date.strftime("%Y-%m-%d"),
                "sku_id": sku,
                "brand": sku_brands[sku],
                "category": sku_cats[sku],
                "quantity": qty,
                "unit_price_inr": price,
                "discount_pct": round(discount * 100, 1),
                "net_revenue_inr": net_price,
                "gross_margin_inr": gross_margin,
                "channel": channel,
                "city": city,
                "customer_segment": seg,
                # BUG-29 fix: only physical retail stores for Offline channel
                # (exclude Online and Spa-only stores which don't sell retail products)
                "store_id": rng.choice(
                    [s[0] for s in STORES if s[4] not in ("Online", "App", "Spa")]
                )
                if channel == "Offline"
                else ("ON001" if channel == "Online" else "ON002"),
            }
        )
    txn_df = pd.DataFrame(txn_rows)
    txn_df.to_csv(OUT / "huft_sales_transactions.csv", index=False)
    print(f"huft_sales_transactions.csv — {len(txn_df):,} transactions")

    # ── 7. huft_returns.csv ───────────────────────────────────────────────────
    # ~3% return rate
    return_reasons = [
        "Product expired",
        "Wrong product sent",
        "Quality issue",
        "Pet didn't like it",
        "Damaged packaging",
        "Customer changed mind",
        "Allergic reaction",
        "Duplicate order",
    ]
    ret_rows = []
    sampled_returns = txn_df.sample(frac=0.03, random_state=42)
    for _, r in sampled_returns.iterrows():
        ret_rows.append(
            {
                "return_id": f"RET_{len(ret_rows) + 1:05d}",
                "original_txn_id": r["txn_id"],
                "sku_id": r["sku_id"],
                "category": r["category"],
                "brand": r["brand"],
                "return_date": (
                    pd.Timestamp(r["date"]) + pd.Timedelta(days=int(rng.integers(1, 8)))
                ).strftime("%Y-%m-%d"),
                "quantity_returned": int(rng.integers(1, r["quantity"] + 1)),
                "return_reason": rng.choice(return_reasons),
                "refund_inr": round(r["net_revenue_inr"] * rng.uniform(0.5, 1.0), 2),
                "channel": r["channel"],
                "city": r["city"],
            }
        )
    ret_df = pd.DataFrame(ret_rows)
    ret_df.to_csv(OUT / "huft_returns.csv", index=False)
    print(f"huft_returns.csv   — {len(ret_df):,} returns")

    # ── 8. huft_supplier_performance.csv ─────────────────────────────────────
    # Build supplier → promised lead time lookup from PRODUCTS list
    # SUPPLIERS_HUFT tuples: (name, email, country, base_otd, quality_rating,
    #                         min_order_qty, is_domestic, notes)
    # There is no lead_time field in the tuple — derive from product definitions.
    _sup_lead: dict[str, float] = {}
    for p in PRODUCTS:
        sup_name = p.get("supplier", "")
        lt = p.get("lead_time_days", 7)
        if sup_name not in _sup_lead:
            _sup_lead[sup_name] = []  # type: ignore[assignment]
        _sup_lead[sup_name].append(lt)  # type: ignore[attr-defined]
    _sup_lead_avg: dict[str, float] = {
        k: float(sum(v) / len(v))
        for k, v in _sup_lead.items()
        if v  # type: ignore[arg-type]
    }

    sp_rows = []
    review_months = pd.date_range("2023-01-01", "2024-12-01", freq="MS")
    for sup in SUPPLIERS_HUFT:
        name = sup[0]
        base_otd = sup[3]
        # Use the average contracted lead time for this supplier from the products
        # definition; fall back to 10 days if the supplier name isn't found.
        promised_lt = _sup_lead_avg.get(name, 10.0)
        for mon in review_months:
            sp_rows.append(
                {
                    "supplier_name": name,
                    "review_month": mon.strftime("%Y-%m"),
                    "on_time_delivery_pct": round(
                        float(np.clip(base_otd + rng.normal(0, 1.5), 70, 100)), 1
                    ),
                    "defect_rate_pct": round(
                        float(np.clip(rng.normal(1.5, 0.8), 0, 10)), 2
                    ),
                    "fill_rate_pct": round(
                        float(np.clip(rng.normal(95, 3), 75, 100)), 1
                    ),
                    "lead_time_actual_days": max(
                        1, int(rng.normal(promised_lt, max(promised_lt * 0.15, 1.0)))
                    ),
                    # BUG-019 fix: sup[4] is quality_rating in SUPPLIERS_HUFT tuple
                    # (name, email, country, base_otd, quality_rating, ...)
                    "quality_rating": round(
                        float(np.clip(float(sup[4]) + rng.normal(0, 0.1), 1, 5)),
                        1,
                    ),
                    "notes": "",
                }
            )
    sp_df = pd.DataFrame(sp_rows)
    sp_df.to_csv(OUT / "huft_supplier_performance.csv", index=False)
    print(f"huft_supplier_performance.csv — {len(sp_df):,} rows")

    # ── 9. huft_cold_chain.csv ───────────────────────────────────────────────
    cold_skus = [p for p in PRODUCTS if p["is_cold_chain"]]
    cold_rows = []
    for sku in cold_skus:
        for d in dates:
            temp = round(float(rng.normal(3.5, 1.2)), 1)  # target: 2-6°C
            expiry = (d + pd.Timedelta(days=int(rng.integers(3, 8)))).strftime(
                "%Y-%m-%d"
            )
            breach = temp < 0 or temp > 8
            cold_rows.append(
                {
                    "date": d.strftime("%Y-%m-%d"),
                    "sku_id": sku["sku_id"],
                    "name": sku["name"],
                    "temp_celsius": temp,
                    "target_min_c": 2.0,
                    "target_max_c": 6.0,
                    "temp_breach": breach,
                    "units_in_cold_storage": int(rng.integers(20, 120)),
                    "expiry_date": expiry,
                    "shelf_life_days_remaining": int(rng.integers(2, 8)),
                    "units_at_risk_of_expiry": int(rng.integers(0, 15))
                    if rng.random() < 0.08
                    else 0,
                    "batch_id": f"BATCH_{sku['sku_id']}_{d.strftime('%Y%m%d')}",
                }
            )
    cold_df = pd.DataFrame(cold_rows)
    cold_df.to_csv(OUT / "huft_cold_chain.csv", index=False)
    print(f"huft_cold_chain.csv — {len(cold_df):,} rows")

    # ── 10. store_daily_inventory.csv — per-store, per-SKU, per-day ──────────
    # Each physical store carries a portion of national inventory.
    # Distribution is based on store size (sqft) weighted by store type.
    # Flagship stores carry ~25% more per sqft than Standard stores.
    # Online channels carry a single pooled inventory.
    # Only last 90 days generated (recent operational data — keeps file manageable).
    print("Generating store_daily_inventory.csv (this may take a minute)...")

    # Filter to physical stores only (not online/spa)
    phys_stores = [s for s in STORES if s[4] not in ("Online", "App", "Spa")]

    # Compute store weight = size_sqft × type_multiplier
    type_mult = {"Flagship": 1.35, "Standard": 1.0, "Express": 0.55, "Spa": 0.3}
    store_weights = {}
    for s in phys_stores:
        store_weights[s[0]] = s[6] * type_mult.get(s[4], 1.0)  # sqft × multiplier
    total_weight = sum(store_weights.values())
    store_share = {sid: w / total_weight for sid, w in store_weights.items()}

    # Build store lookup
    store_meta = {
        s[0]: {
            "city": s[1],
            "state": s[2],
            "region": s[3],
            "store_type": s[4],
            "has_spa": s[7],
        }
        for s in phys_stores
    }

    # Last 90 days only
    last_90_dates = dates[-90:]

    sdi_rows = []
    for sku in PRODUCTS:
        sku_id = sku["sku_id"]
        # National demand/inventory for this SKU over last 90 days
        sku_national = (
            demand_df[demand_df["sku_id"] == sku_id].tail(90).reset_index(drop=True)
        )
        if sku_national.empty:
            continue

        for store_id, share in store_share.items():
            meta = store_meta[store_id]

            # Skip spa stores for non-spa products
            if meta["has_spa"] is False and sku["category"] == "Grooming Consumables":
                continue

            # Per-store demand = national demand × share + small local noise
            for i, row in sku_national.iterrows():
                store_demand = max(
                    0, int(round(float(row["demand"]) * share * rng.normal(1.0, 0.15)))
                )
                store_inv = max(
                    0,
                    int(round(float(row["inventory"]) * share * rng.normal(1.0, 0.12))),
                )
                lead = sku["lead_time_days"]
                # BUG-8 fix: use actual store_demand (not static base_demand)
                # so DoS reflects real daily demand including seasonal variation.
                # Use max(1, ...) to avoid DOS=infinity when store_demand is 0 (rare).
                avg_d_actual = max(store_demand, 1)
                dos = round(store_inv / avg_d_actual, 1)
                if dos < lead:
                    risk = "CRITICAL"
                elif dos < lead * 2:
                    risk = "WARNING"
                else:
                    risk = "OK"

                sdi_rows.append(
                    {
                        "date": row["date"],
                        "store_id": store_id,
                        "city": meta["city"],
                        "state": meta["state"],
                        "region": meta["region"],
                        "store_type": meta["store_type"],
                        "sku_id": sku_id,
                        "name": sku["name"],
                        "category": sku["category"],
                        "brand": sku["brand"],
                        "demand": store_demand,
                        "inventory": store_inv,
                        "lead_time_days": lead,
                        "days_of_supply": dos,
                        "risk_status": risk,
                        "price_inr": sku["price_inr"],
                        "cost_inr": sku["cost_inr"],
                    }
                )

    sdi_df = pd.DataFrame(sdi_rows)
    sdi_df.to_csv(OUT / "store_daily_inventory.csv", index=False)
    print(
        f"store_daily_inventory.csv — {len(sdi_df):,} rows "
        f"({len(phys_stores)} stores × {len(PRODUCTS)} SKUs × 90 days)"
    )

    # ── 11. Backward-compat: keep pet_store_supply_chain.csv ─────────────────
    compat = demand_df.copy()
    compat["price_usd"] = compat["price_inr"]
    compat["region"] = "India"
    compat = compat[
        [
            "date",
            "sku_id",
            "name",
            "category",
            "subcategory",
            "supplier",
            "region",
            "demand",
            "inventory",
            "lead_time_days",
            "price_usd",
        ]
    ]
    compat.to_csv(OUT / "pet_store_supply_chain.csv", index=False)
    print(f"pet_store_supply_chain.csv — {len(compat):,} rows (backward compat)")

    return demand_df


if __name__ == "__main__":
    generate()
