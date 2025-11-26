#!/usr/bin/env python3
"""
Anemia Dataset Analysis
Analysis of positive vs negative anemia cases in the Eyes-defy-anemia dataset
"""

import pandas as pd
import numpy as np

def main():
    # Read the Excel files
    india_path = "/Users/ekas/Desktop/Anemia_Detection/dataset anemia/India/India.xlsx"
    italy_path = "/Users/ekas/Desktop/Anemia_Detection/dataset anemia/Italy/Italy.xlsx"

    print("📊 ANEMIA DATASET ANALYSIS")
    print("=" * 60)
    
    # Load data
    india_df = pd.read_excel(india_path)
    italy_df = pd.read_excel(italy_path)

    # Clean the Italy dataset
    italy_clean = italy_df[['Number', 'Hgb', 'Gender', 'Age', 'Note']].copy()

    # Fix comma decimal separators and convert to numeric
    italy_clean['Hgb_clean'] = italy_clean['Hgb'].astype(str).str.replace(',', '.')
    italy_clean['Hgb_clean'] = pd.to_numeric(italy_clean['Hgb_clean'], errors='coerce')

    # Remove invalid entries
    italy_clean = italy_clean[italy_clean['Hgb_clean'].notna()].copy()
    italy_clean['Hgb'] = italy_clean['Hgb_clean']

    # Define anemia thresholds (WHO standards)
    def classify_anemia(hgb, gender):
        """Classify anemia based on WHO standards"""
        if pd.isna(hgb) or pd.isna(gender):
            return 'Unknown'
        if gender == 'M':
            return 'Anemic' if hgb < 13.0 else 'Non-anemic'
        elif gender == 'F':
            return 'Anemic' if hgb < 12.0 else 'Non-anemic'
        else:
            return 'Unknown'

    # Apply classification
    india_df['Anemia_Status'] = india_df.apply(lambda row: classify_anemia(row['Hgb'], row['Gender']), axis=1)
    italy_clean['Anemia_Status'] = italy_clean.apply(lambda row: classify_anemia(row['Hgb'], row['Gender']), axis=1)

    # Combine datasets
    combined_df = pd.concat([
        india_df[['Number', 'Hgb', 'Gender', 'Age', 'Anemia_Status']].assign(Country='India'),
        italy_clean[['Number', 'Hgb', 'Gender', 'Age', 'Anemia_Status']].assign(Country='Italy')
    ], ignore_index=True)

    # Overall statistics
    total_samples = len(combined_df)
    total_anemic = len(combined_df[combined_df['Anemia_Status'] == 'Anemic'])
    total_non_anemic = len(combined_df[combined_df['Anemia_Status'] == 'Non-anemic'])

    print(f"\n🌍 OVERALL DATASET SUMMARY")
    print(f"Total samples: {total_samples}")
    print(f"✅ POSITIVE for anemia: {total_anemic} ({total_anemic/total_samples*100:.1f}%)")
    print(f"❌ NEGATIVE for anemia: {total_non_anemic} ({total_non_anemic/total_samples*100:.1f}%)")

    # Individual dataset analysis
    print(f"\n🇮🇳 INDIA DATASET (95 samples)")
    india_anemic = len(india_df[india_df['Anemia_Status'] == 'Anemic'])
    india_non_anemic = len(india_df[india_df['Anemia_Status'] == 'Non-anemic'])
    print(f"   ✅ Positive: {india_anemic} ({india_anemic/len(india_df)*100:.1f}%)")
    print(f"   ❌ Negative: {india_non_anemic} ({india_non_anemic/len(india_df)*100:.1f}%)")
    print(f"   📊 Hgb range: {india_df['Hgb'].min():.1f} - {india_df['Hgb'].max():.1f} g/dL (mean: {india_df['Hgb'].mean():.2f})")

    print(f"\n🇮🇹 ITALY DATASET (122 samples after cleaning)")
    italy_anemic = len(italy_clean[italy_clean['Anemia_Status'] == 'Anemic'])
    italy_non_anemic = len(italy_clean[italy_clean['Anemia_Status'] == 'Non-anemic'])
    print(f"   ✅ Positive: {italy_anemic} ({italy_anemic/len(italy_clean)*100:.1f}%)")
    print(f"   ❌ Negative: {italy_non_anemic} ({italy_non_anemic/len(italy_clean)*100:.1f}%)")
    print(f"   📊 Hgb range: {italy_clean['Hgb'].min():.1f} - {italy_clean['Hgb'].max():.1f} g/dL (mean: {italy_clean['Hgb'].mean():.2f})")

    # Gender breakdown
    print(f"\n👥 GENDER ANALYSIS")
    print("Combined dataset by gender:")
    gender_breakdown = combined_df.groupby(['Gender', 'Anemia_Status']).size().unstack(fill_value=0)
    print(gender_breakdown)
    
    # Calculate gender-specific rates
    female_total = len(combined_df[combined_df['Gender'] == 'F'])
    female_anemic = len(combined_df[(combined_df['Gender'] == 'F') & (combined_df['Anemia_Status'] == 'Anemic')])
    male_total = len(combined_df[combined_df['Gender'] == 'M'])
    male_anemic = len(combined_df[(combined_df['Gender'] == 'M') & (combined_df['Anemia_Status'] == 'Anemic')])
    
    print(f"\n👩 Female anemia rate: {female_anemic}/{female_total} = {female_anemic/female_total*100:.1f}%")
    print(f"👨 Male anemia rate: {male_anemic}/{male_total} = {male_anemic/male_total*100:.1f}%")

    # Age analysis
    print(f"\n📈 AGE ANALYSIS")
    anemic_ages = combined_df[combined_df['Anemia_Status'] == 'Anemic']['Age'].dropna()
    non_anemic_ages = combined_df[combined_df['Anemia_Status'] == 'Non-anemic']['Age'].dropna()
    
    print(f"Anemic patients - Mean age: {anemic_ages.mean():.1f} years (range: {anemic_ages.min():.0f}-{anemic_ages.max():.0f})")
    print(f"Non-anemic patients - Mean age: {non_anemic_ages.mean():.1f} years (range: {non_anemic_ages.min():.0f}-{non_anemic_ages.max():.0f})")

    # Hemoglobin analysis
    print(f"\n🩸 HEMOGLOBIN LEVEL ANALYSIS")
    anemic_hgb = combined_df[combined_df['Anemia_Status'] == 'Anemic']['Hgb']
    non_anemic_hgb = combined_df[combined_df['Anemia_Status'] == 'Non-anemic']['Hgb']
    
    print(f"Anemic patients - Mean Hgb: {anemic_hgb.mean():.2f} g/dL (range: {anemic_hgb.min():.1f}-{anemic_hgb.max():.1f})")
    print(f"Non-anemic patients - Mean Hgb: {non_anemic_hgb.mean():.2f} g/dL (range: {non_anemic_hgb.min():.1f}-{non_anemic_hgb.max():.1f})")

    # WHO thresholds reminder
    print(f"\n⚕️  WHO ANEMIA THRESHOLDS USED:")
    print(f"   • Adult Males: < 13.0 g/dL")
    print(f"   • Adult Females: < 12.0 g/dL")

    print(f"\n🎯 DATASET UTILITY FOR ML/AI:")
    print(f"   • Class balance: {min(total_anemic, total_non_anemic)/max(total_anemic, total_non_anemic):.2f} (closer to 1.0 = better)")
    print(f"   • Sufficient positive cases: {'Yes' if total_anemic > 50 else 'Limited'}")
    print(f"   • Cross-population validation: {'Available (India vs Italy)' if len(india_df) > 0 and len(italy_clean) > 0 else 'No'}")

if __name__ == "__main__":
    main()