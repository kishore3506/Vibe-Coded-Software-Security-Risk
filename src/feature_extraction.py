"""
Feature Extraction Pipeline for Source Code Security Risk Analysis.
Extracts code metrics, security-sensitive indicators, and TF-IDF token representations.
Treats code strictly as text: NO eval, NO exec, NO dynamic execution.
"""

import re
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Regular expression patterns for tokenization and metrics
WORD_TOKEN_PATTERN = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b')

# Security-sensitive keywords and API lists
MEM_ALLOC_PATTERNS = [r'\bmalloc\b', r'\bcalloc\b', r'\brealloc\b', r'\bfree\b', r'\bnew\b', r'\bdelete\b']
BUFFER_UNSAFE_PATTERNS = [
    r'\bstrcpy\b', r'\bstrcat\b', r'\bgets\b', r'\bsprintf\b', r'\bvsprintf\b',
    r'\bscanf\b', r'\bsscanf\b', r'\bmemcpy\b', r'\bmemmove\b', r'\bstrncpy\b',
    r'\bstrncat\b', r'\bsnprintf\b'
]
FORMAT_STRING_PATTERNS = [r'%s', r'%n', r'%x', r'%d', r'\bprintf\s*\(\s*[^",]+,\s*']
COMMAND_EXEC_PATTERNS = [r'\bsystem\s*\(', r'\bpopen\s*\(', r'\bexec[lv]?[pe]?\s*\(', r'\bfork\s*\(', r'\bspawn\b', r'\bsubprocess\b']
FILE_IO_PATTERNS = [r'\bfopen\b', r'\bfread\b', r'\bfwrite\b', r'\bfclose\b', r'\bopen\s*\(', r'\bread\s*\(', r'\bwrite\s*\(']
SECRETS_PATTERNS = [r'password\b', r'passwd\b', r'secret\b', r'api_?key\b', r'token\b', r'credential\b', r'private_?key\b']
ERROR_HANDLING_PATTERNS = [r'\berrno\b', r'\bperror\b', r'\bassert\b', r'\btry\b', r'\bcatch\b', r'\bthrow\b']

def extract_code_metrics(code: str) -> Dict[str, float]:
    """
    Extract structural, lexical, and security-oriented numerical metrics from raw source code.
    Pure static text analysis.
    """
    if not isinstance(code, str) or not code.strip():
        # Empty code fallback
        return {
            "char_count": 0.0,
            "line_count": 0.0,
            "avg_line_length": 0.0,
            "blank_line_count": 0.0,
            "comment_line_count": 0.0,
            "comment_ratio": 0.0,
            "max_indent_depth": 0.0,
            "num_tokens": 0.0,
            "num_digits": 0.0,
            "if_count": 0.0,
            "else_count": 0.0,
            "for_count": 0.0,
            "while_count": 0.0,
            "switch_count": 0.0,
            "case_count": 0.0,
            "return_count": 0.0,
            "goto_count": 0.0,
            "total_branching": 0.0,
            "pointer_deref_count": 0.0,
            "array_subscript_count": 0.0,
            "bitwise_op_count": 0.0,
            "logical_op_count": 0.0,
            "relational_op_count": 0.0,
            "arithmetic_op_count": 0.0,
            "assignment_op_count": 0.0,
            "mem_alloc_count": 0.0,
            "buffer_unsafe_count": 0.0,
            "format_string_count": 0.0,
            "command_exec_count": 0.0,
            "file_io_count": 0.0,
            "secrets_count": 0.0,
            "null_check_count": 0.0,
            "cast_count": 0.0,
            "sizeof_count": 0.0,
            "error_handling_count": 0.0
        }

    lines = code.splitlines()
    line_count = len(lines)
    char_count = len(code)
    avg_line_length = char_count / max(1, line_count)

    blank_lines = sum(1 for line in lines if not line.strip())
    comment_lines = sum(1 for line in lines if line.strip().startswith(("//", "/*", "*", "#")))
    comment_ratio = comment_lines / max(1, line_count)

    # Indentation depth
    max_indent = 0
    for line in lines:
        if line.strip():
            leading_spaces = len(line) - len(line.lstrip(' '))
            leading_tabs = len(line) - len(line.lstrip('\t'))
            depth = leading_tabs + (leading_spaces // 4)
            if depth > max_indent:
                max_indent = depth

    tokens = WORD_TOKEN_PATTERN.findall(code)
    num_tokens = len(tokens)
    num_digits = sum(c.isdigit() for c in code)

    # Control flow & branching
    lower_code = code.lower()
    if_count = len(re.findall(r'\bif\b', lower_code))
    else_count = len(re.findall(r'\belse\b', lower_code))
    for_count = len(re.findall(r'\bfor\b', lower_code))
    while_count = len(re.findall(r'\bwhile\b', lower_code))
    switch_count = len(re.findall(r'\bswitch\b', lower_code))
    case_count = len(re.findall(r'\bcase\b', lower_code))
    return_count = len(re.findall(r'\breturn\b', lower_code))
    goto_count = len(re.findall(r'\bgoto\b', lower_code))
    total_branching = if_count + else_count + for_count + while_count + switch_count + case_count

    # Operators
    pointer_deref = code.count("->") + code.count("*")
    array_subscript = code.count("[")
    bitwise_ops = code.count("&") + code.count("|") + code.count("^") + code.count("~") + code.count("<<") + code.count(">>")
    logical_ops = code.count("&&") + code.count("||") + code.count("!")
    relational_ops = code.count("==") + code.count("!=") + code.count("<=") + code.count(">=")
    arithmetic_ops = code.count("+") + code.count("-") + code.count("/") + code.count("%")
    assignment_ops = code.count("=")

    # Security indicators
    mem_alloc = sum(len(re.findall(p, code, re.IGNORECASE)) for p in MEM_ALLOC_PATTERNS)
    buffer_unsafe = sum(len(re.findall(p, code, re.IGNORECASE)) for p in BUFFER_UNSAFE_PATTERNS)
    format_strings = sum(len(re.findall(p, code)) for p in FORMAT_STRING_PATTERNS)
    command_exec = sum(len(re.findall(p, code, re.IGNORECASE)) for p in COMMAND_EXEC_PATTERNS)
    file_io = sum(len(re.findall(p, code, re.IGNORECASE)) for p in FILE_IO_PATTERNS)
    secrets = sum(len(re.findall(p, code, re.IGNORECASE)) for p in SECRETS_PATTERNS)
    null_checks = len(re.findall(r'\b(NULL|nullptr|None)\b', code))
    casts = len(re.findall(r'\([a-zA-Z_][a-zA-Z0-9_]*\s*\*?\)', code))
    sizeof_count = len(re.findall(r'\bsizeof\b', code))
    error_handling = sum(len(re.findall(p, code, re.IGNORECASE)) for p in ERROR_HANDLING_PATTERNS)

    return {
        "char_count": float(char_count),
        "line_count": float(line_count),
        "avg_line_length": float(avg_line_length),
        "blank_line_count": float(blank_lines),
        "comment_line_count": float(comment_lines),
        "comment_ratio": float(comment_ratio),
        "max_indent_depth": float(max_indent),
        "num_tokens": float(num_tokens),
        "num_digits": float(num_digits),
        "if_count": float(if_count),
        "else_count": float(else_count),
        "for_count": float(for_count),
        "while_count": float(while_count),
        "switch_count": float(switch_count),
        "case_count": float(case_count),
        "return_count": float(return_count),
        "goto_count": float(goto_count),
        "total_branching": float(total_branching),
        "pointer_deref_count": float(pointer_deref),
        "array_subscript_count": float(array_subscript),
        "bitwise_op_count": float(bitwise_ops),
        "logical_op_count": float(logical_ops),
        "relational_op_count": float(relational_ops),
        "arithmetic_op_count": float(arithmetic_ops),
        "assignment_op_count": float(assignment_ops),
        "mem_alloc_count": float(mem_alloc),
        "buffer_unsafe_count": float(buffer_unsafe),
        "format_string_count": float(format_strings),
        "command_exec_count": float(command_exec),
        "file_io_count": float(file_io),
        "secrets_count": float(secrets),
        "null_check_count": float(null_checks),
        "cast_count": float(casts),
        "sizeof_count": float(sizeof_count),
        "error_handling_count": float(error_handling)
    }

def detect_security_patterns(code: str) -> List[Dict[str, str]]:
    """
    Identifies rule-based static security pattern indicators.
    NOTE: These are heuristic indicators designed for screening and are DISTINCT
    from the Artificial Neural Network model's learned representation.
    """
    indicators = []
    if not isinstance(code, str) or not code.strip():
        return indicators

    # 1. Unsafe Buffer Manipulation
    unsafe_funcs = [f for f in ['strcpy', 'strcat', 'gets', 'sprintf', 'vsprintf'] if re.search(r'\b' + f + r'\b', code)]
    if unsafe_funcs:
        indicators.append({
            "category": "CWE-120 / CWE-787: Buffer Overflow Vulnerability",
            "severity": "HIGH",
            "details": f"Detected legacy unsafe string/memory function(s): {', '.join(unsafe_funcs)}. These lack bounds checking and may allow out-of-bounds memory writes."
        })

    # 2. Arbitrary Command Execution
    exec_funcs = [f for f in ['system', 'popen', 'fork', 'exec', 'subprocess'] if re.search(r'\b' + f + r'\b', code)]
    if exec_funcs:
        indicators.append({
            "category": "CWE-78: OS Command Injection Risk",
            "severity": "HIGH",
            "details": f"Detected system command execution invocation: {', '.join(exec_funcs)}. If inputs are not sanitized, arbitrary OS commands can be executed."
        })

    # 3. Format String Vulnerabilities
    if re.search(r'\bprintf\s*\(\s*[^",]+\s*\)', code) or re.search(r'\bsprintf\s*\(\s*[^,]+,\s*[^",]+\s*\)', code):
        indicators.append({
            "category": "CWE-134: Uncontrolled Format String",
            "severity": "HIGH",
            "details": "Potential format string vulnerability: variable passed directly as format specifier to printf/sprintf without static string constant."
        })

    # 4. Hardcoded Credentials
    if re.search(r'(password|passwd|secret|api_key|token)\s*=\s*["\'][^"\']{4,}["\']', code, re.IGNORECASE):
        indicators.append({
            "category": "CWE-798: Use of Hard-coded Credentials",
            "severity": "MEDIUM",
            "details": "Possible hard-coded secret, password, or API key detected in source code literal assignment."
        })

    # 5. Memory Management & Pointer Safety
    has_free = bool(re.search(r'\bfree\s*\(', code))
    has_malloc = bool(re.search(r'\b(malloc|calloc|realloc)\s*\(', code))
    if has_free and has_malloc:
        indicators.append({
            "category": "CWE-401 / CWE-416: Dynamic Memory Lifecycle Management",
            "severity": "LOW",
            "details": "Manual heap allocation and deallocation present. Verify null checks after allocation and zeroing pointers post-free to prevent Use-After-Free."
        })

    # 6. Unbounded Input Reading
    if re.search(r'\bgets\s*\(', code):
        indicators.append({
            "category": "CWE-242: Use of Inherently Dangerous Function (gets)",
            "severity": "CRITICAL",
            "details": "gets() does not check for buffer overrun and has been removed from modern C standards. Replace with fgets()."
        })

    return indicators

def fit_tfidf_vectorizer(train_texts: List[str], max_features: int = 50) -> TfidfVectorizer:
    """
    Fits a TF-IDF vectorizer strictly on training source code texts.
    Extracts high-frequency code tokens and syntax structures.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        token_pattern=r'\b[a-zA-Z_][a-zA-Z0-9_]*\b',
        ngram_range=(1, 2),
        lowercase=True
    )
    vectorizer.fit(train_texts)
    return vectorizer

def build_feature_dataframe(codes: List[str], tfidf_vectorizer: TfidfVectorizer = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Combines extracted numerical metrics with TF-IDF features into a single feature matrix.
    Returns:
        DataFrame of features, and list of feature column names.
    """
    metrics_list = [extract_code_metrics(c) for c in codes]
    metrics_df = pd.DataFrame(metrics_list)
    metric_cols = list(metrics_df.columns)

    if tfidf_vectorizer is not None:
        tfidf_sparse = tfidf_vectorizer.transform(codes)
        tfidf_feature_names = [f"tfidf_{w}" for w in tfidf_vectorizer.get_feature_names_out()]
        tfidf_df = pd.DataFrame(tfidf_sparse.toarray(), columns=tfidf_feature_names)
        combined_df = pd.concat([metrics_df, tfidf_df], axis=1)
    else:
        combined_df = metrics_df

    all_cols = list(combined_df.columns)
    return combined_df, all_cols
