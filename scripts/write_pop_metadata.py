import csv
from cyvcf2 import VCF
demo_names = ["const", "bottleneck_2000", "exp", "step_05", "step_005","step_0005","split_500","split_1000","split_2000","out_of_africa"]
for i in range(0,10):
    vcf_path = f'official_data2/diploid_sim/{demo_names[i]}/{demo_names[i]}.vcf.gz'
    meta_path = f'official_data2/diploid_sim/{demo_names[i]}/{demo_names[i]}.meta.csv'
    
    samples = VCF(vcf_path).samples  # guaranteed to match VCF column order
    print(f"{demo_names[i]}")
    
    with open(meta_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["population", "name"])
        for name in samples:
            parts = name.split("_")
            if i in range(3,6):
                pop =  "_".join(parts[1:3])
            else:
                pop = parts[1] # -> "D0"
            writer.writerow([pop, name])
