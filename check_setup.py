"""
Setup Verification Script
Checks that all required packages are properly installed
and API keys are accessible via python-dotenv.
"""
import sys
import os
from dotenv import load_dotenv

def test_imports():
    print("-" * 50)
    print("Testing installed packages...")
    print("-" * 50)
    
    packages = [
        ("langchain", "LangChain core"),
        ("langchain_community", "LangChain Community"),
        ("langchain_google_genai", "LangChain Google GenAI"),
        ("langchain_groq", "LangChain Groq"),
        ("faiss", "FAISS Vector DB"),
        ("pypdf", "PyPDF Loader"),
        ("dotenv", "Python Dotenv")
    ]
    
    all_passed = True
    for module_name, display_name in packages:
        try:
            __import__(module_name)
            print(f"[OK] {display_name} successfully imported.")
        except ImportError as e:
            print(f"[FAILED] Could not import {display_name} ({module_name}): {e}")
            all_passed = False
            
    return all_passed

def test_environment_variables():
    print("\n" + "-" * 50)
    print("Testing environment variables (.env)...")
    print("-" * 50)
    
    # Load environment variables from .env file
    load_dotenv()
    
    gemini_key = os.getenv("GOOGLE_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    status = True
    
    if gemini_key and gemini_key != "your_gemini_api_key_here":
        # Mask the key for display security
        masked_key = gemini_key[:6] + "..." + gemini_key[-4:]
        print(f"[OK] GOOGLE_API_KEY detected: {masked_key}")
    else:
        print("[WARNING] GOOGLE_API_KEY is missing or still set to placeholder in .env")
        status = False
        
    if groq_key and groq_key != "your_groq_api_key_here":
        masked_key = groq_key[:6] + "..." + groq_key[-4:]
        print(f"[OK] GROQ_API_KEY detected: {masked_key}")
    else:
        print("[WARNING] GROQ_API_KEY is missing or still set to placeholder in .env")
        status = False
        
    return status

if __name__ == "__main__":
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {sys.version.split()[0]}\n")
    
    imports_ok = test_imports()
    env_ok = test_environment_variables()
    
    print("\n" + "=" * 50)
    if imports_ok and env_ok:
        print("ALL CHECKS PASSED! Ready for Step 2.")
    elif imports_ok and not env_ok:
        print("PACKAGES OK, but please configure your API keys in .env.")
    else:
        print("SETUP FAILED. Please resolve package installation errors.")
    print("=" * 50)