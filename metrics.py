from typing import List, Set, Dict
import numpy as np

def recall_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    topk = recommended[:k]
    hits = sum(1 for r in topk if r in relevant)
    return hits / len(relevant)

def average_precision_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for i, rec in enumerate(recommended[:k], start=1):
        if rec in relevant:
            hits += 1
            score += hits / i
    return score / min(len(relevant), k)

def mean_recall_at_k(all_recs: Dict[str, List[str]], all_rels: Dict[str, Set[str]], k: int) -> float:
    return np.mean([recall_at_k(all_recs[q], all_rels[q], k) for q in all_recs])

def map_at_k(all_recs: Dict[str, List[str]], all_rels: Dict[str, Set[str]], k: int) -> float:
    return np.mean([average_precision_at_k(all_recs[q], all_rels[q], k) for q in all_recs])

if __name__ == "__main__":
    all_recs = {
    "icici_admin": [
        "basic-computer-literacy-windows-10-new",  
        "administrative-professional-short-form", 
        "financial-professional-short-form",       
        "bank-administrative-assistant-short-form",
        "verify-numerical-ability",                
        "excel-basics",                            
        "general-entry-level-data-entry-7-0-solution"  
    ],
    "java_dev_needs": [
        "automata-fix-new",                        
        "core-java-entry-level-new",               
        "networking-essentials",                   
        "java-8-new",                              
        "core-java-advanced-level-new",            
        "agile-software-development",              
        "technology-professional-8-0-job-focused-assessment",  
        "computer-science-new"                    
    ],
    "sales_grads": [
        "entry-level-sales-7-1",                   
        "entry-level-sales-sift-out-7-1",          
        "general-entry-level-data-entry-7-0-solution",  
        "sales-representative-solution",           
        "technical-sales-associate-solution",      
        "sales-support-specialist-solution",       
        "english-comprehension-new",               
        "svar-spoken-english-indian-accent-new",   
        "sales-and-service-phone-simulation",      
        "sales-and-service-phone-solution"        
    ],
    "content_writer": [
        "drupal-new",                             
        "search-engine-optimization-new",          
        "entry-level-sales-sift-out-7-1",          
        "administrative-professional-short-form",  
        "communication-skills-test"            
    ],
    "coo_china": [
        "global-skills-assessment",                
        "graduate-8-0-job-focused-assessment",     
        "personality-profiling-tool",             
        "motivation-questionnaire-mqm5",           
        "executive-leadership-test"                
    ]
}


    all_rels = {
        "icici_admin": {
            "basic-computer-literacy-windows-10-new",
            "verify-numerical-ability",
            "administrative-professional-short-form",
            "financial-professional-short-form",
            "bank-administrative-assistant-short-form",
            "general-entry-level-data-entry-7-0-solution"
        },
        "java_dev_needs": {
            "automata-fix-new",
            "core-java-entry-level-new",
            "java-8-new",
            "core-java-advanced-level-new",
            "agile-software-development",
            "technology-professional-8-0-job-focused-assessment",
            "computer-science-new"
        },
        "sales_grads": {
            "entry-level-sales-7-1",
            "entry-level-sales-sift-out-7-1",
            "entry-level-sales-solution",
            "sales-representative-solution",
            "sales-support-specialist-solution",
            "technical-sales-associate-solution",
            "svar-spoken-english-indian-accent-new",
            "sales-and-service-phone-solution",
            "sales-and-service-phone-simulation",
            "english-comprehension-new"
        },
        "content_writer": {
            "drupal-new",
            "search-engine-optimization-new"
        },
        "coo_china": {
            "motivation-questionnaire-mqm5",
            "global-skills-assessment",
            "graduate-8-0-job-focused-assessment"
        }
    }

    for K in (3, 5, 10):
        print(f"\n====== Metrics @K={K} ======")
        print(f"Mean Recall@{K}: {mean_recall_at_k(all_recs, all_rels, K):.4f}")
        print(f"MAP@{K}:        {map_at_k(all_recs, all_rels, K):.4f}")
