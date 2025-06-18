#!/bin/bash
# run_mmlu_sequential.sh - Run MMLU benchmarks sequentially to avoid download freezes
# This script runs each MMLU subject as a separate evaluation to prevent timeouts

# Default parameters
MODEL_PATH="$HOME/Models/ANE/anemll-Llama-3.2-1B-FP16-b64-ctx1024"
LIMIT=10
NUM_SHOTS=0
SAFETY_MARGIN=100  # Default safety margin
DOWNLOAD_TIMEOUT=180
MAX_RETRIES=5
DEBUG=""

# Get current date for folder naming
CURRENT_DATE=$(date +"%Y%m%d")
MODEL_NAME=$(basename "$MODEL_PATH")
OUTPUT_DIR="evaluate/results/mmlu_${MODEL_NAME}_${CURRENT_DATE}"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            MODEL_NAME=$(basename "$MODEL_PATH")
            OUTPUT_DIR="evaluate/results/mmlu_${MODEL_NAME}_${CURRENT_DATE}"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --shots)
            NUM_SHOTS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --download-timeout)
            DOWNLOAD_TIMEOUT="$2"
            shift 2
            ;;
        --max-retries)
            MAX_RETRIES="$2"
            shift 2
            ;;
        --safety-margin)
            SAFETY_MARGIN="$2"
            shift 2
            ;;
        --debug)
            DEBUG="--debug"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Define MMLU subject categories and weights
# Using simple arrays instead of associative arrays for better compatibility
STEM_SUBJECTS=(
    "mmlu_abstract_algebra"
    "mmlu_astronomy"
    "mmlu_college_biology"
    "mmlu_college_chemistry" 
    "mmlu_college_computer_science"
    "mmlu_college_mathematics"
    "mmlu_college_physics"
    "mmlu_computer_security"
    "mmlu_conceptual_physics"
    "mmlu_electrical_engineering"
    "mmlu_elementary_mathematics"
    "mmlu_high_school_biology"
    "mmlu_high_school_chemistry"
)

HUMANITIES_SUBJECTS=(
    "mmlu_formal_logic"
    "mmlu_high_school_european_history"
    "mmlu_high_school_us_history"
    "mmlu_high_school_world_history"
    "mmlu_international_law"
    "mmlu_jurisprudence"
    "mmlu_philosophy"
    "mmlu_world_religions"
)

SOCIAL_SCIENCES_SUBJECTS=(
    "mmlu_econometrics"
    "mmlu_high_school_government_and_politics"
    "mmlu_high_school_geography"
    "mmlu_high_school_macroeconomics"
    "mmlu_high_school_microeconomics"
    "mmlu_high_school_psychology"
    "mmlu_human_sexuality"
    "mmlu_professional_psychology"
    "mmlu_public_relations"
    "mmlu_security_studies"
    "mmlu_sociology"
)

MEDICAL_SUBJECTS=(
    "mmlu_anatomy"
    "mmlu_clinical_knowledge"
    "mmlu_college_medicine"
    "mmlu_human_aging"
    "mmlu_medical_genetics"
    "mmlu_nutrition"
    "mmlu_professional_medicine"
    "mmlu_virology"
)

# Define combined list for baseline subjects (commonly used subset)
MMLU_SUBJECTS=(
    "mmlu_abstract_algebra"
    "mmlu_anatomy"
    "mmlu_astronomy" 
    "mmlu_business_ethics"
    "mmlu_clinical_knowledge"
    "mmlu_college_biology"
    "mmlu_college_chemistry"
    "mmlu_college_computer_science"
    "mmlu_college_mathematics"
    "mmlu_college_medicine"
    "mmlu_college_physics"
    "mmlu_computer_security"
    "mmlu_conceptual_physics"
    "mmlu_econometrics"
    "mmlu_electrical_engineering"
    "mmlu_elementary_mathematics"
    "mmlu_formal_logic"
    "mmlu_global_facts"
    "mmlu_high_school_biology"
    "mmlu_high_school_chemistry"
)

# Function to check if a subject belongs to a category
is_in_category() {
    local subject="$1"
    local category="$2"
    
    case "$category" in
        "STEM")
            for s in "${STEM_SUBJECTS[@]}"; do
                if [ "$s" = "$subject" ]; then
                    return 0
                fi
            done
            ;;
        "Humanities")
            for s in "${HUMANITIES_SUBJECTS[@]}"; do
                if [ "$s" = "$subject" ]; then
                    return 0
                fi
            done
            ;;
        "Social Sciences")
            for s in "${SOCIAL_SCIENCES_SUBJECTS[@]}"; do
                if [ "$s" = "$subject" ]; then
                    return 0
                fi
            done
            ;;
        "Medical")
            for s in "${MEDICAL_SUBJECTS[@]}"; do
                if [ "$s" = "$subject" ]; then
                    return 0
                fi
            done
            ;;
    esac
    
    return 1
}

# Function to get a subject's category
get_category() {
    local subject="$1"
    
    if is_in_category "$subject" "STEM"; then
        echo "STEM"
    elif is_in_category "$subject" "Humanities"; then
        echo "Humanities"
    elif is_in_category "$subject" "Social Sciences"; then
        echo "Social Sciences"
    elif is_in_category "$subject" "Medical"; then
        echo "Medical"
    else
        echo "Other"
    fi
}

# Function to get category weight
get_category_weight() {
    local category="$1"
    
    case "$category" in
        "STEM")
            echo "0.3"
            ;;
        "Humanities")
            echo "0.25"
            ;;
        "Social Sciences")
            echo "0.25"
            ;;
        "Medical")
            echo "0.2"
            ;;
        *)
            echo "0.1"
            ;;
    esac
}

# Function to run a single MMLU subject
run_mmlu_subject() {
    local subject=$1
    echo ""
    echo "========================================================"
    echo "Running evaluation for $subject"
    local category=$(get_category "$subject")
    echo "Category: $category"
    echo "========================================================"
    
    # Create subject-specific output directory
    local subject_dir="$OUTPUT_DIR/$subject"
    mkdir -p "$subject_dir"
    
    # Run the evaluation script
    ./run_eval.sh \
        --model "$MODEL_PATH" \
        --tasks "$subject" \
        --limit $LIMIT \
        --shots $NUM_SHOTS \
        --output-dir "$subject_dir" \
        --download-timeout $DOWNLOAD_TIMEOUT \
        --max-retries $MAX_RETRIES \
        --safety-margin $SAFETY_MARGIN \
        $DEBUG
    
    # Check if the evaluation was successful
    if [ $? -eq 0 ]; then
        # Copy the JSON file to a more descriptive name
        if [ -f "$subject_dir/results.json" ]; then
            cp "$subject_dir/results.json" "$subject_dir/${subject}.json"
        fi
        echo "Successfully evaluated $subject"
        return 0
    else
        echo "Failed to evaluate $subject"
        return 1
    fi
}

# Function to aggregate results with weighted scoring
aggregate_results() {
    echo ""
    echo "========================================================"
    echo "Aggregating results from all subjects"
    echo "========================================================"
    
    # Create a combined results file
    COMBINED_FILE="$OUTPUT_DIR/mmlu_combined_${MODEL_NAME}_${CURRENT_DATE}.json"
    
    # Initialize JSON structure
    echo "{" > $COMBINED_FILE
    echo "  \"model\": \"${MODEL_NAME}\"," >> $COMBINED_FILE
    echo "  \"date\": \"${CURRENT_DATE}\"," >> $COMBINED_FILE
    echo "  \"shots\": ${NUM_SHOTS}," >> $COMBINED_FILE
    echo "  \"examples_per_subject\": ${LIMIT}," >> $COMBINED_FILE
    echo "  \"subject_scores\": {" >> $COMBINED_FILE
    
    # Variables for tracking overall and category performance
    total_score=0
    total_subjects=0
    
    # Initialize category trackers
    stem_score=0
    stem_count=0
    humanities_score=0
    humanities_count=0
    social_sciences_score=0
    social_sciences_count=0
    medical_score=0
    medical_count=0
    other_score=0
    other_count=0
    
    # Track all subjects with scores for easier analysis
    declare -a subject_list=()
    declare -a score_list=()
    declare -a category_list=()
    
    # Iterate through all subjects and extract results
    first=true
    for subject in "${MMLU_SUBJECTS[@]}"; do
        # Check if results exist for this subject
        if [ -f "$OUTPUT_DIR/$subject/results.json" ]; then
            # Extract accuracy score
            acc=$(grep "acc,none" "$OUTPUT_DIR/$subject/results.json" | head -n 1 | sed 's/.*: \([0-9.]*\).*/\1/')
            
            if [ -n "$acc" ]; then
                # Store the score
                subject_list+=("$subject")
                score_list+=("$acc")
                
                # Get category for this subject
                category=$(get_category "$subject")
                category_list+=("$category")
                
                # Add to category total
                case "$category" in
                    "STEM")
                        stem_score=$(echo "$stem_score + $acc" | bc)
                        stem_count=$((stem_count + 1))
                        ;;
                    "Humanities")
                        humanities_score=$(echo "$humanities_score + $acc" | bc)
                        humanities_count=$((humanities_count + 1))
                        ;;
                    "Social Sciences")
                        social_sciences_score=$(echo "$social_sciences_score + $acc" | bc)
                        social_sciences_count=$((social_sciences_count + 1))
                        ;;
                    "Medical")
                        medical_score=$(echo "$medical_score + $acc" | bc)
                        medical_count=$((medical_count + 1))
                        ;;
                    *)
                        other_score=$(echo "$other_score + $acc" | bc)
                        other_count=$((other_count + 1))
                        ;;
                esac
                
                # Add to overall total
                total_score=$(echo "$total_score + $acc" | bc)
                ((total_subjects++))
                
                # Format for combined file
                if [ "$first" = true ]; then
                    first=false
                else
                    echo "," >> $COMBINED_FILE
                fi
                
                # Write to combined file - strip the mmlu_ prefix for cleaner output
                clean_name="${subject#mmlu_}"
                echo "    \"$clean_name\": {" >> $COMBINED_FILE
                echo "      \"accuracy\": $acc," >> $COMBINED_FILE
                echo "      \"category\": \"$category\"" >> $COMBINED_FILE
                echo "    }" >> $COMBINED_FILE
            fi
        fi
    done
    
    # Close subject scores section
    echo "  }," >> $COMBINED_FILE
    
    # Calculate category averages
    echo "  \"category_scores\": {" >> $COMBINED_FILE
    
    # Add STEM category
    if [ $stem_count -gt 0 ]; then
        stem_avg=$(echo "scale=4; $stem_score / $stem_count" | bc)
        echo "    \"STEM\": {" >> $COMBINED_FILE
        echo "      \"accuracy\": $stem_avg," >> $COMBINED_FILE
        echo "      \"subjects\": $stem_count," >> $COMBINED_FILE
        echo "      \"weight\": 0.3" >> $COMBINED_FILE
        echo "    }" >> $COMBINED_FILE
    fi
    
    # Add Humanities category
    if [ $humanities_count -gt 0 ]; then
        [ $stem_count -gt 0 ] && echo "," >> $COMBINED_FILE
        humanities_avg=$(echo "scale=4; $humanities_score / $humanities_count" | bc)
        echo "    \"Humanities\": {" >> $COMBINED_FILE
        echo "      \"accuracy\": $humanities_avg," >> $COMBINED_FILE
        echo "      \"subjects\": $humanities_count," >> $COMBINED_FILE
        echo "      \"weight\": 0.25" >> $COMBINED_FILE
        echo "    }" >> $COMBINED_FILE
    fi
    
    # Add Social Sciences category
    if [ $social_sciences_count -gt 0 ]; then
        [ $humanities_count -gt 0 -o $stem_count -gt 0 ] && echo "," >> $COMBINED_FILE
        social_avg=$(echo "scale=4; $social_sciences_score / $social_sciences_count" | bc)
        echo "    \"Social_Sciences\": {" >> $COMBINED_FILE
        echo "      \"accuracy\": $social_avg," >> $COMBINED_FILE
        echo "      \"subjects\": $social_sciences_count," >> $COMBINED_FILE
        echo "      \"weight\": 0.25" >> $COMBINED_FILE
        echo "    }" >> $COMBINED_FILE
    fi
    
    # Add Medical category
    if [ $medical_count -gt 0 ]; then
        [ $social_sciences_count -gt 0 -o $humanities_count -gt 0 -o $stem_count -gt 0 ] && echo "," >> $COMBINED_FILE
        medical_avg=$(echo "scale=4; $medical_score / $medical_count" | bc)
        echo "    \"Medical\": {" >> $COMBINED_FILE
        echo "      \"accuracy\": $medical_avg," >> $COMBINED_FILE
        echo "      \"subjects\": $medical_count," >> $COMBINED_FILE
        echo "      \"weight\": 0.2" >> $COMBINED_FILE
        echo "    }" >> $COMBINED_FILE
    fi
    
    # Add Other category
    if [ $other_count -gt 0 ]; then
        [ $medical_count -gt 0 -o $social_sciences_count -gt 0 -o $humanities_count -gt 0 -o $stem_count -gt 0 ] && echo "," >> $COMBINED_FILE
        other_avg=$(echo "scale=4; $other_score / $other_count" | bc)
        echo "    \"Other\": {" >> $COMBINED_FILE
        echo "      \"accuracy\": $other_avg," >> $COMBINED_FILE
        echo "      \"subjects\": $other_count," >> $COMBINED_FILE
        echo "      \"weight\": 0.1" >> $COMBINED_FILE
        echo "    }" >> $COMBINED_FILE
    fi
    
    # Close category scores section
    echo "  }," >> $COMBINED_FILE
    
    # Calculate standard average
    avg_score=0
    if [ $total_subjects -gt 0 ]; then
        avg_score=$(echo "scale=4; $total_score / $total_subjects" | bc)
    fi
    
    # Calculate weighted score
    weighted_score=0
    total_weight=0
    
    # Add weights for categories with subjects
    if [ $stem_count -gt 0 ]; then
        stem_avg=$(echo "scale=4; $stem_score / $stem_count" | bc)
        weighted_score=$(echo "$weighted_score + $stem_avg * 0.3" | bc)
        total_weight=$(echo "$total_weight + 0.3" | bc)
    fi
    
    if [ $humanities_count -gt 0 ]; then
        humanities_avg=$(echo "scale=4; $humanities_score / $humanities_count" | bc)
        weighted_score=$(echo "$weighted_score + $humanities_avg * 0.25" | bc)
        total_weight=$(echo "$total_weight + 0.25" | bc)
    fi
    
    if [ $social_sciences_count -gt 0 ]; then
        social_avg=$(echo "scale=4; $social_sciences_score / $social_sciences_count" | bc)
        weighted_score=$(echo "$weighted_score + $social_avg * 0.25" | bc)
        total_weight=$(echo "$total_weight + 0.25" | bc)
    fi
    
    if [ $medical_count -gt 0 ]; then
        medical_avg=$(echo "scale=4; $medical_score / $medical_count" | bc)
        weighted_score=$(echo "$weighted_score + $medical_avg * 0.2" | bc)
        total_weight=$(echo "$total_weight + 0.2" | bc)
    fi
    
    if [ $other_count -gt 0 ]; then
        other_avg=$(echo "scale=4; $other_score / $other_count" | bc)
        weighted_score=$(echo "$weighted_score + $other_avg * 0.1" | bc)
        total_weight=$(echo "$total_weight + 0.1" | bc)
    fi
    
    # Normalize if needed
    if (( $(echo "$total_weight > 0" | bc -l) )); then
        weighted_score=$(echo "scale=4; $weighted_score / $total_weight" | bc)
    fi
    
    # Add averages to combined file
    echo "  \"total_subjects\": $total_subjects," >> $COMBINED_FILE
    echo "  \"mmlu_average\": $avg_score," >> $COMBINED_FILE
    echo "  \"mmlu_weighted_average\": $weighted_score" >> $COMBINED_FILE
    
    # Close the JSON file
    echo "}" >> $COMBINED_FILE
    
    # Print summary
    echo "Combined results saved to $COMBINED_FILE"
    echo "Summary:"
    echo "  Total subjects evaluated: $total_subjects"
    
    # Print category summaries
    [ $stem_count -gt 0 ] && echo "  STEM average: $(echo "scale=4; $stem_score / $stem_count" | bc) ($stem_count subjects, weight: 0.3)"
    [ $humanities_count -gt 0 ] && echo "  Humanities average: $(echo "scale=4; $humanities_score / $humanities_count" | bc) ($humanities_count subjects, weight: 0.25)"
    [ $social_sciences_count -gt 0 ] && echo "  Social Sciences average: $(echo "scale=4; $social_sciences_score / $social_sciences_count" | bc) ($social_sciences_count subjects, weight: 0.25)"
    [ $medical_count -gt 0 ] && echo "  Medical average: $(echo "scale=4; $medical_score / $medical_count" | bc) ($medical_count subjects, weight: 0.2)"
    [ $other_count -gt 0 ] && echo "  Other average: $(echo "scale=4; $other_score / $other_count" | bc) ($other_count subjects, weight: 0.1)"
    
    echo "  Unweighted average: $avg_score"
    echo "  Weighted average: $weighted_score"
    
    # Generate readable report
    REPORT_FILE="$OUTPUT_DIR/mmlu_report_${MODEL_NAME}_${CURRENT_DATE}.txt"
    {
        echo "MMLU Evaluation Report"
        echo "======================="
        echo "Model: $MODEL_NAME"
        echo "Date: $CURRENT_DATE"
        echo "Examples per subject: $LIMIT"
        echo "Number of shots: $NUM_SHOTS"
        echo "Total subjects evaluated: $total_subjects"
        echo ""
        
        echo "Category Summaries:"
        # Print STEM category
        if [ $stem_count -gt 0 ]; then
            stem_avg=$(echo "scale=4; $stem_score / $stem_count" | bc)
            echo "  STEM: $stem_avg ($stem_count subjects, weight: 0.3)"
            
            # Print subjects in this category
            echo "    Subjects:"
            for i in "${!subject_list[@]}"; do
                if [ "${category_list[$i]}" = "STEM" ]; then
                    echo "      - ${subject_list[$i]#mmlu_}: ${score_list[$i]}"
                fi
            done
            echo ""
        fi
        
        # Print Humanities category
        if [ $humanities_count -gt 0 ]; then
            humanities_avg=$(echo "scale=4; $humanities_score / $humanities_count" | bc)
            echo "  Humanities: $humanities_avg ($humanities_count subjects, weight: 0.25)"
            
            # Print subjects in this category
            echo "    Subjects:"
            for i in "${!subject_list[@]}"; do
                if [ "${category_list[$i]}" = "Humanities" ]; then
                    echo "      - ${subject_list[$i]#mmlu_}: ${score_list[$i]}"
                fi
            done
            echo ""
        fi
        
        # Print Social Sciences category
        if [ $social_sciences_count -gt 0 ]; then
            social_avg=$(echo "scale=4; $social_sciences_score / $social_sciences_count" | bc)
            echo "  Social Sciences: $social_avg ($social_sciences_count subjects, weight: 0.25)"
            
            # Print subjects in this category
            echo "    Subjects:"
            for i in "${!subject_list[@]}"; do
                if [ "${category_list[$i]}" = "Social Sciences" ]; then
                    echo "      - ${subject_list[$i]#mmlu_}: ${score_list[$i]}"
                fi
            done
            echo ""
        fi
        
        # Print Medical category
        if [ $medical_count -gt 0 ]; then
            medical_avg=$(echo "scale=4; $medical_score / $medical_count" | bc)
            echo "  Medical: $medical_avg ($medical_count subjects, weight: 0.2)"
            
            # Print subjects in this category
            echo "    Subjects:"
            for i in "${!subject_list[@]}"; do
                if [ "${category_list[$i]}" = "Medical" ]; then
                    echo "      - ${subject_list[$i]#mmlu_}: ${score_list[$i]}"
                fi
            done
            echo ""
        fi
        
        # Print Other category
        if [ $other_count -gt 0 ]; then
            other_avg=$(echo "scale=4; $other_score / $other_count" | bc)
            echo "  Other: $other_avg ($other_count subjects, weight: 0.1)"
            
            # Print subjects in this category
            echo "    Subjects:"
            for i in "${!subject_list[@]}"; do
                if [ "${category_list[$i]}" = "Other" ]; then
                    echo "      - ${subject_list[$i]#mmlu_}: ${score_list[$i]}"
                fi
            done
            echo ""
        fi
        
        echo "Overall Results:"
        echo "  Unweighted average: $avg_score"
        echo "  Weighted average: $weighted_score"
    } > "$REPORT_FILE"
    
    echo "Detailed report saved to $REPORT_FILE"
}

# Print run configuration
echo "========================================================"
echo "Running MMLU subjects sequentially"
echo "========================================================"
echo "Model path: $MODEL_PATH"
echo "Model name: $MODEL_NAME"
echo "Subjects: ${#MMLU_SUBJECTS[@]} common MMLU subjects"
echo "Limit per subject: $LIMIT examples"
echo "Number of shots: $NUM_SHOTS"
echo "Output directory: $OUTPUT_DIR"
echo "Download timeout: $DOWNLOAD_TIMEOUT seconds"
echo "Max retries: $MAX_RETRIES"
echo "Safety margin: $SAFETY_MARGIN tokens"
echo "========================================================"

# Run each subject
success_count=0
for subject in "${MMLU_SUBJECTS[@]}"; do
    run_mmlu_subject "$subject"
    if [ $? -eq 0 ]; then
        ((success_count++))
    fi
done

# Aggregate results
aggregate_results

echo ""
echo "========================================================"
echo "MMLU Sequential Evaluation Complete"
echo "========================================================"
echo "Successfully evaluated $success_count/${#MMLU_SUBJECTS[@]} subjects"
echo "Results saved to $OUTPUT_DIR"
echo "Detailed report available at $REPORT_FILE"
echo "========================================================" 