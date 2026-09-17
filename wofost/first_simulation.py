import pcse
import pandas as pd
import matplotlib.pyplot as plt



# 1. Start the built-in WOFOST demonstration


wofost = pcse.start_wofost()

print("WOFOST model:")
print(type(wofost))



# 2. Run the complete crop simulation


wofost.run_till_terminate()



# 3. Extract daily simulation output


output = wofost.get_output()

print(f"Number of simulation days: {len(output)}")



# 4. Convert output to Pandas DataFrame


df = pd.DataFrame(output)

print("\nWOFOST output columns:")
print(df.columns.tolist())

print("\nFirst five records:")
print(df.head())



# 5. Save simulation output


df.to_csv("outputs/wofost_output.csv", index=False)

print("\nOutput saved to outputs/wofost_output.csv")



# 6. Plot LAI


plt.figure()
plt.plot(df["day"], df["LAI"])
plt.xlabel("Date")
plt.ylabel("LAI")
plt.title("WOFOST Simulated LAI")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/lai_plot.png", dpi=300)
plt.show()



# 7. Plot TAGP


plt.figure()
plt.plot(df["day"], df["TAGP"])
plt.xlabel("Date")
plt.ylabel("TAGP")
plt.title("WOFOST Simulated Total Above-Ground Biomass")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/tagp_plot.png", dpi=300)
plt.show()



# 8. Plot TWSO


plt.figure()
plt.plot(df["day"], df["TWSO"])
plt.xlabel("Date")
plt.ylabel("TWSO")
plt.title("WOFOST Simulated Storage-Organ Biomass")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/twso_plot.png", dpi=300)
plt.show()



# 9. Get seasonal summary


summary = wofost.get_summary_output()

print("\nWOFOST seasonal summary:")
print(summary)