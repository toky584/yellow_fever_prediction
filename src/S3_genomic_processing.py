# src/S3_genomic_processing.py
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from Bio import SeqIO
from collections import Counter

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture # BayesianGaussianMixture also an option
from utils import normalize_text
# from gensim.models import Word2Vec # Uncomment if using Word2Vec k-mer embeddings
from utils import haversine_vectorized # If needed for spatial matching to main data

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

METADATA_FILE = RAW_DATA_DIR / "sequences.xlsx"
FASTA_FILE = RAW_DATA_DIR / "sequences.fasta"
# Input from S2 (data with spillover info + environmental vars)
S2_ENV_MERGED_DATA_FILE = PROCESSED_DATA_DIR / "S2_data_with_environment.csv"

# Output files from this script
GENOMIC_METADATA_WITH_CLUSTERS_FILE = PROCESSED_DATA_DIR / "S3_genomic_metadata_with_clusters.csv"
OUTPUT_FINAL_DATA_WITH_GENOMIC_FILE = PROCESSED_DATA_DIR / "S3_final_data_with_genomic_clusters.csv"

# K-mer settings
KMER_K_VALUE = 2 
KMER_PCA_COMPONENTS = 5 

# DBSCAN parameters (from your notebook, choose a good one or make configurable)
DBSCAN_EPS = 1.4 # Example from notebook pg 34, you found others via grid search
DBSCAN_MIN_SAMPLES = 52 # Example

# --- Genomic Feature Functions (from notebook pages 16-17) ---
def nucleotide_composition(sequence_str):
    """Calculates nucleotide composition (A, T/U, C, G) for a sequence string."""
    sequence_str = sequence_str.upper()
    length = len(sequence_str)
    if length == 0:
        return np.array([0.0, 0.0, 0.0, 0.0])
    # Handle both DNA and RNA by checking for U
    count_t_or_u = sequence_str.count('T') + sequence_str.count('U')
    return np.array([
        sequence_str.count('A') / length,
        count_t_or_u / length,
        sequence_str.count('C') / length,
        sequence_str.count('G') / length
    ])

def kmer_features(sequences_list, k_val):
    """Converts a list of DNA/RNA sequence strings into k-mer frequency vectors."""
    kmer_counts_list_of_dicts = []
    for seq_str in sequences_list:
        seq_str = seq_str.upper()
        if len(seq_str) < k_val:
            kmer_counts_list_of_dicts.append(Counter()) # Empty counter for short sequences
            continue
        kmers = [seq_str[i:i + k_val] for i in range(len(seq_str) - k_val + 1)]
        kmer_counts_list_of_dicts.append(Counter(kmers))

    if not kmer_counts_list_of_dicts: # Handle empty input sequence list
        return np.array([]).reshape(0,0) # Return empty 2D array

    vectorizer = DictVectorizer(sparse=False)
    # Fit_transform might fail if all sequences are too short leading to empty counters for all
    try:
        kmer_matrix = vectorizer.fit_transform(kmer_counts_list_of_dicts)
        return kmer_matrix, vectorizer.feature_names_ # Also return feature names for inspection
    except ValueError as e:
        print(f"Error during k-mer vectorization (possibly all sequences too short or all counters empty): {e}")
        # Try to find how many features would have been generated if at least one sequence was valid
        temp_vec = DictVectorizer(sparse=False)
        try:
            # Try fitting on a dummy valid kmer to get feature count
            temp_vec.fit([{'AA':1}]) # Use a k-mer of length k_val
            num_potential_features = len(temp_vec.feature_names_)
            return np.zeros((len(sequences_list), num_potential_features)), None # Return zeros if vectorization fails
        except: # Fallback if even dummy fails
             return np.array([]).reshape(len(sequences_list),0), None


def gc_content(sequence_str):
    """Calculates GC content of a sequence string."""
    sequence_str = sequence_str.upper()
    length = len(sequence_str)
    if length == 0:
        return 0.0
    gc_val = sequence_str.count('G') + sequence_str.count('C')
    return gc_val / length

# --- Word2Vec k-mer Embeddings (Optional)---
# def train_kmer_word2vec(sequences_list, k=6, vector_size=100, window=5, min_count=1, sg=1):
#     print(f"Training Word2Vec for {k}-mers...")
#     corpus = [[seq_str[i:i+k] for i in range(len(seq_str)-k+1)] for seq_str in sequences_list if len(seq_str) >= k]
#     if not corpus:
#         print("Warning: No sequences long enough for Word2Vec k-mer corpus.")
#         return None
#     model = Word2Vec(corpus, vector_size=vector_size, window=window, min_count=min_count, sg=sg, workers=4)
#     return model

# def get_sequence_embeddings_word2vec(sequences_list, w2v_model, k=6):
#     print("Generating Word2Vec k-mer embeddings...")
#     embeddings = []
#     if w2v_model is None:
#         print("Word2Vec model not available, returning zero embeddings.")
#         # Need to know vector_size to return zeros of correct shape
#         # Assuming it's a parameter or can be inferred if model was trained.
#         # For now, this path would likely lead to an error later if not handled.
#         # It's better to ensure w2v_model exists or handle this case explicitly in combine_features.
#         return np.array([np.zeros(100) for _ in sequences_list]) # Placeholder if vector_size is 100

#     for seq_str in sequences_list:
#         if len(seq_str) < k:
#             embeddings.append(np.zeros(w2v_model.vector_size))
#             continue
#         kmers_in_seq = [seq_str[i:i+k] for i in range(len(seq_str)-k+1)]
#         vecs = [w2v_model.wv[kmer] for kmer in kmers_in_seq if kmer in w2v_model.wv]
#         if vecs:
#             embeddings.append(np.mean(vecs, axis=0))
#         else:
#             embeddings.append(np.zeros(w2v_model.vector_size))
#     return np.array(embeddings)

# --- Clustering Evaluation Function (from notebook page 35-36) ---
def evaluate_genomic_clustering(metadata_df_with_labels, title_prefix,
                                host_col='Host_new', cluster_label_col='cluster_result'):
    """Evaluates clustering based on Host information (Human vs. Primate)."""
    # Ensure 'Host_new' column exists
    if host_col not in metadata_df_with_labels.columns:
        print(f"Warning: Host column '{host_col}' not found in metadata for evaluating '{title_prefix}'.")
        return

    eval_df = metadata_df_with_labels[metadata_df_with_labels[host_col].isin(['Human', 'Primate'])].copy()
    if eval_df.empty:
        print(f"No 'Human' or 'Primate' samples found for evaluating '{title_prefix}'.")
        return

    print(f"\n--- Clustering Evaluation: {title_prefix} ---")
    unique_clusters = sorted(eval_df[cluster_label_col].unique())

    for cluster_id in unique_clusters:
        cluster_data = eval_df[eval_df[cluster_label_col] == cluster_id]
        human_in_cluster = len(cluster_data[cluster_data[host_col] == 'Human'])
        primate_in_cluster = len(cluster_data[cluster_data[host_col] == 'Primate'])
        total_human = len(eval_df[eval_df[host_col] == 'Human'])
        total_primate = len(eval_df[eval_df[host_col] == 'Primate'])

        print(f"  Cluster {cluster_id}:")
        if total_human > 0:
            print(f"    Human samples in this cluster: {human_in_cluster} ({human_in_cluster/total_human:.2%})")
        if total_primate > 0:
            print(f"    Primate samples in this cluster: {primate_in_cluster} ({primate_in_cluster/total_primate:.2%})")
    print("--- End Evaluation ---")


# --- Main Execution ---
def main():
    print("--- S3: Starting Genomic Data Processing and Clustering ---")

    # 1. Load Metadata and FASTA sequences
    try:
        metadata_raw = pd.read_excel(METADATA_FILE)
        fasta_records_dict = {record.id: str(record.seq) for record in SeqIO.parse(FASTA_FILE, "fasta")}
        print(f"Loaded {len(metadata_raw)} metadata records and {len(fasta_records_dict)} FASTA sequences.")
    except FileNotFoundError as e:
        print(f"Error: Metadata or FASTA file not found. {e}")
        return
    except Exception as e:
        print(f"Error loading sequence data: {e}")
        return

    # 2. Filter Metadata 
    metadata = metadata_raw[
        (metadata_raw['Org_location'] == 'Brazil') &
        (metadata_raw['Geo_Location'].notna()) &
        (metadata_raw['Geo_Location'].str.contains('Brazil', case=False, na=False))
    ].copy()
    print(f"Metadata after initial filters (Brazil, Geo_Location): {len(metadata)} records.")

    # Parse release_date and sort
    metadata['release_date'] = pd.to_datetime(metadata['Release_Date'], errors='coerce')
    metadata.dropna(subset=['release_date'], inplace=True) # Remove rows where date couldn't be parsed
    metadata['month'] = metadata['release_date'].dt.month
    metadata['year'] = metadata['release_date'].dt.year
    metadata = metadata.sort_values(by=['year', 'month'])

    # 3. Filter out Vector Hosts (notebook page 16)
    all_species_in_metadata = set(metadata['Host'].dropna().unique())
    
    # From notebook pg 16
    vectors_set = {
        'Aedes albopictus', 'Anopheles evansae', 'Culex', 'Haemagogus',
        'Haemagogus janthinomys', 'Haemagogus leucocelaenus',
        'Ochlerotatus argyrothorax', 'Ochlerotatus scapularis',
        'Ochlerotatus taeniorhynchus', 'Sabethes chloropterus',
        'Uranotaenia pulcherrima'
    }
    # Create non_vectors by removing vector species from all unique host species found in metadata
    non_vector_hosts_filter = [species for species in all_species_in_metadata if species not in vectors_set]
    
    metadata_non_vectors = metadata[metadata['Host'].isin(non_vector_hosts_filter)].copy()
    print(f"Metadata after filtering vector hosts: {len(metadata_non_vectors)} records.")
    if metadata_non_vectors.empty:
        print("No non-vector host metadata remaining. Exiting.")
        return

    # 4. Extract Sequences for Filtered Metadata (notebook page 17)
    sequences_for_analysis = []
    metadata_with_sequences = [] # To store metadata rows that have a sequence
    inexistant_accessions = []

    for _, row in metadata_non_vectors.iterrows():
        accession = str(row['Accession']).strip()
        # NCBI accessions might have versions like .1, .2. FASTA IDs might or might not.
        # Try matching common patterns.
        seq_found = fasta_records_dict.get(accession)
        if not seq_found and '.' not in accession: # If base accession and not found, try with .1
             seq_found = fasta_records_dict.get(accession + ".1")
        # Add more sophisticated matching if needed (e.g. regex or splitting version numbers)

        if seq_found:
            sequences_for_analysis.append(seq_found)
            metadata_with_sequences.append(row.to_dict()) # Store as dict to rebuild DF later
        else:
            inexistant_accessions.append(accession)
            
    if inexistant_accessions:
        print(f"Warning: {len(inexistant_accessions)} sequences not found for accessions like: {inexistant_accessions[:5]}")
    
    if not sequences_for_analysis:
        print("No sequences extracted for analysis. Exiting.")
        return
        
    processed_metadata_df = pd.DataFrame(metadata_with_sequences)
    print(f"Extracted {len(sequences_for_analysis)} sequences for {len(processed_metadata_df)} metadata records.")

    # 5. Assign 'Host_new' (Human, Primate, Unknown) (notebook page 18)
    def assign_host_new(host_val_series):
        # Host categories (expand as needed based on your actual 'Host' column values)
        human_markers = ['Homo sapiens']
        primate_markers = ['Alouatta', 'Callicebus', 'Callithrix', 'Cebus', 
                           'Saguinus', 'Sapajus', 'Simiiformes', 'Primates', 'marmosets'] # Add all relevant
        
        conditions = [
            host_val_series.isin(human_markers),
            host_val_series.apply(lambda x: any(pm in str(x) for pm in primate_markers) if pd.notna(x) else False)
        ]
        choices = ['Human', 'Primate']
        return np.select(conditions, choices, default='Unknown')

    processed_metadata_df['Host_new'] = assign_host_new(processed_metadata_df['Host'])
    print("Host distribution ('Host_new'):")
    print(processed_metadata_df['Host_new'].value_counts())

    # 6. Feature Extraction 
    print("Extracting genomic features...")
    gc_feat = np.array([gc_content(s) for s in sequences_for_analysis]).reshape(-1, 1)
    freq_feat = np.array([nucleotide_composition(s) for s in sequences_for_analysis])
    kmer_feat_raw, kmer_names = kmer_features(sequences_for_analysis, k_val=KMER_K_VALUE)

    if kmer_feat_raw.size == 0 : # Check if kmer_features returned an empty useful array
        print("K-mer feature extraction failed or yielded no features. Using only GC and Freq.")
        if gc_feat.shape[0] == 0 or freq_feat.shape[0] == 0:
             print("GC or Freq features also empty. Cannot proceed.")
             return
        combined_features_for_clustering = np.hstack([freq_feat, gc_feat])
        kmer_feat_pca = np.empty((len(sequences_for_analysis), 0)) # Placeholder
    elif kmer_feat_raw.shape[1] < KMER_PCA_COMPONENTS:
        print(f"K-mer features ({kmer_feat_raw.shape[1]}) less than PCA components ({KMER_PCA_COMPONENTS}). Using raw k-mers.")
        kmer_feat_pca = kmer_feat_raw
        combined_features_for_clustering = np.hstack([kmer_feat_pca, freq_feat, gc_feat])
    else:
        pca = PCA(n_components=KMER_PCA_COMPONENTS)
        kmer_feat_pca = pca.fit_transform(kmer_feat_raw)
        combined_features_for_clustering = np.hstack([kmer_feat_pca, freq_feat, gc_feat])

    print(f"Shapes -> GC: {gc_feat.shape}, Freq: {freq_feat.shape}, Kmer (raw): {kmer_feat_raw.shape}, Kmer (PCA): {kmer_feat_pca.shape}")
    print(f"Combined features for clustering shape: {combined_features_for_clustering.shape}")

    if combined_features_for_clustering.shape[0] == 0:
        print("No combined features to process. Exiting clustering.")
        return

    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(combined_features_for_clustering)

    # 7. Unsupervised Clustering 
    print("\nPerforming Unsupervised Clustering...")
    cluster_results = {}

    kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
    cluster_results['KMeans'] = kmeans.fit_predict(normalized_features)

    dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    cluster_results['DBSCAN'] = dbscan.fit_predict(normalized_features)

    gmm = GaussianMixture(n_components=2, random_state=42, covariance_type='full') # full is often more robust
    cluster_results['GaussianMixture'] = gmm.fit_predict(normalized_features)

    agglo = AgglomerativeClustering(n_clusters=2, linkage='ward')
    cluster_results['Agglomerative'] = agglo.fit_predict(normalized_features)

    try:
        # SpectralClustering can be sensitive to graph connectivity
        spectral = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
                                      assign_labels='kmeans', random_state=42)
        cluster_results['Spectral'] = spectral.fit_predict(normalized_features)
    except Exception as e:
        print(f"Spectral clustering failed: {e}. Filling with -2 (error).")
        cluster_results['Spectral'] = np.full(len(normalized_features), -2)

    # Evaluate and store cluster labels in metadata
    for algo_name, labels in cluster_results.items():
        processed_metadata_df[f'cluster_{algo_name}'] = labels
        evaluate_genomic_clustering(processed_metadata_df, algo_name, cluster_label_col=f'cluster_{algo_name}')
    
    if 'lat' not in processed_metadata_df.columns or 'lon' not in processed_metadata_df.columns:
        print("Warning: 'lat'/'lon' not directly in genomic metadata. Attempting to parse from 'Geo_Location' or merge.")
        try:
            coords_df = pd.read_csv(RAW_DATA_DIR / "municipalities_coordinates.csv") # Adjust filename
            coords_df['Municipality_norm'] = coords_df['Municipality'].apply(normalize_text) # Assuming 'Municipality' column
            
            # Attempt to extract a comparable key from Geo_Location
            # This is highly dependent on Geo_Location string format
            def extract_region_for_coords(geo_loc_str):
                if pd.isna(geo_loc_str): return None
                parts = geo_loc_str.split(':')
                if len(parts) > 1:
                    region_part = parts[1].split('-')[0].strip() # E.g., "Macae" from "Brazil: Macae-RJ"
                    return normalize_text(region_part)
                return normalize_text(geo_loc_str) # Fallback

            processed_metadata_df['Region_for_coords'] = processed_metadata_df['Geo_Location'].apply(extract_region_for_coords)
            
            # Merge - this is tricky if names don't align perfectly
            processed_metadata_df = pd.merge(processed_metadata_df, coords_df[['Municipality_norm', 'Latitude', 'Longitude']],
                                     left_on='Region_for_coords', right_on='Municipality_norm', how='left')
            processed_metadata_df.rename(columns={'Latitude':'lat', 'Longitude':'lon'}, inplace=True)
            # processed_metadata_df.drop(columns=['Region_for_coords', 'Municipality_norm'], inplace=True, errors='ignore')
            if processed_metadata_df['lat'].isnull().any():
                 print(f"Warning: {processed_metadata_df['lat'].isnull().sum()} genomic records still missing coordinates after merge.")
        except FileNotFoundError:
            print("Warning: municipalities_coordinates.csv not found. Coordinates for genomic data might be missing.")
            if 'lat' not in processed_metadata_df.columns: processed_metadata_df['lat'] = np.nan
            if 'lon' not in processed_metadata_df.columns: processed_metadata_df['lon'] = np.nan


    processed_metadata_df.to_csv(GENOMIC_METADATA_WITH_CLUSTERS_FILE, index=False)
    print(f"\nS3: Genomic metadata with cluster labels saved to {GENOMIC_METADATA_WITH_CLUSTERS_FILE}")

    # 8. Merge selected genomic cluster label back to main spillover dataset (from S2)
    try:
        main_data_df = pd.read_csv(S2_ENV_MERGED_DATA_FILE)
    except FileNotFoundError:
        print(f"Error: S2 output file not found at {S2_ENV_MERGED_DATA_FILE}. Cannot add genomic cluster labels.")
        return

    # Select the cluster label to merge 
    # Ensure 'year', 'month', 'lat', 'lon' are present in processed_metadata_df for matching
    genomic_clusters_to_merge = processed_metadata_df[['year', 'month', 'lat', 'lon', 'cluster_GaussianMixture']].copy()
    genomic_clusters_to_merge.rename(columns={'cluster_GaussianMixture': 'cluster_labels'}, inplace=True) # Standard name
    genomic_clusters_to_merge.dropna(subset=['year', 'month', 'lat', 'lon'], inplace=True)

    if genomic_clusters_to_merge.empty:
        print("No genomic cluster data available to merge (empty or missing required columns). Adding empty 'cluster_labels'.")
        main_data_df['cluster_labels'] = -1 # Default if no genomic data
    else:
        print("Merging genomic cluster labels into main dataset (nearest in space for same year)...")
        main_data_df['cluster_labels'] = -1 # Initialize with a default (e.g., -1 for no match)

        # Ensure main_data_df has 'Year', 'lat', 'long'
        if not all(c in main_data_df.columns for c in ['Year', 'lat', 'long']):
            print("Error: main_data_df missing 'Year', 'lat', or 'long'. Cannot merge genomic clusters.")
        else:
            for index, row in tqdm(main_data_df.iterrows(), total=len(main_data_df), desc="Assigning Genomic Clusters"):
                target_lat_main = row['lat']
                target_lon_main = row['long']
                year_main = row['Year']

                # Filter genomic samples for the same year
                relevant_genomic_samples_for_year = genomic_clusters_to_merge[
                    genomic_clusters_to_merge['year'] == year_main
                ].copy()

                if not relevant_genomic_samples_for_year.empty:
                    if 'lat' in relevant_genomic_samples_for_year.columns and 'lon' in relevant_genomic_samples_for_year.columns:
                        # Calculate squared Euclidean distance for speed
                        relevant_genomic_samples_for_year['distance_sq'] = (
                            (relevant_genomic_samples_for_year['lat'] - target_lat_main)**2 +
                            (relevant_genomic_samples_for_year['lon'] - target_lon_main)**2
                        )
                        if not relevant_genomic_samples_for_year['distance_sq'].empty:
                            nearest_genomic_idx = relevant_genomic_samples_for_year['distance_sq'].idxmin()
                            main_data_df.loc[index, 'cluster_labels'] = relevant_genomic_samples_for_year.loc[nearest_genomic_idx, 'cluster_labels']
                    else:
                        if index == 0 : print("Warning: 'lat' or 'lon' missing in genomic_clusters_to_merge for spatial matching.")
                # else: No genomic samples for that year, cluster_labels remains -1

    main_data_df.to_csv(OUTPUT_FINAL_DATA_WITH_GENOMIC_FILE, index=False)
    print(f"S3: Main data merged with genomic cluster labels saved to {OUTPUT_FINAL_DATA_WITH_GENOMIC_FILE}")
    print("--- S3: Finished ---")

if __name__ == "__main__":
    main()