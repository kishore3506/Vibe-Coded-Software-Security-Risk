"""
Security and Safety Tests.
Verifies input length limits, denial-of-service resilience,
and ensures submitted code is never executed dynamically.
"""

import os
import pytest
from src.predict import predict_security_risk, MAX_CODE_LENGTH

def test_oversized_input_rejection():
    # Construct a string exceeding 50,000 characters
    oversized_code = "int a = 1;\n" * 5500
    assert len(oversized_code) > MAX_CODE_LENGTH
    res = predict_security_risk(oversized_code)
    assert "error" in res
    assert res["prediction"] == "OVERSIZED_INPUT"
    assert "exceeds maximum allowable length" in res["error"]

def test_no_dynamic_execution():
    # Pass dangerous code designed to touch a file if executed
    canary_file = os.path.join(os.path.dirname(__file__), "hacked_canary.txt")
    if os.path.exists(canary_file):
        os.remove(canary_file)

    dangerous_payload = f"""
    # Malicious attempt to create canary file
    import os
    os.system("echo exploited > {canary_file}")
    open("{canary_file}", "w").write("pwned")
    eval("__import__('os').system('exit 1')")
    """
    
    # Run analysis
    res = predict_security_risk(dangerous_payload)

    # Strictly assert that the canary file was NEVER created
    assert not os.path.exists(canary_file), "CRITICAL SECURITY BREACH: Submitted code was dynamically executed!"
    assert res["status"] == "success"

def test_shell_metacharacters_passive_handling():
    shell_injection_code = """
    void exec_handler() {
        char *cmd = "; rm -rf / ; cat /etc/passwd ; echo 'test' &";
        system(cmd);
    }
    """
    res = predict_security_risk(shell_injection_code)
    assert res["status"] == "success"
    # Should flag command injection pattern
    categories = [p["category"] for p in res["pattern_indicators"]]
    assert any("Command Injection" in c for c in categories)
