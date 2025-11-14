input_file = "../input_data/NASA_Global_Seawater_Oxygen-18_Database.csv"
output_file = "../input_data/NASA_Global_Seawater_Oxygen-18_Database_clean.csv"

expected_cols = 10

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        parts = line.strip().split(",")
        if len(parts) > expected_cols:
            # Join everything from the 10th item onward into one quoted field
            fixed_ref = ",".join(parts[9:])
            line = ",".join(parts[:9] + [f'"{fixed_ref}"']) + "\n"
        fout.write(line)