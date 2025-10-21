import requests


def check_icd_link(code):
    # Construct the URL for checking the existence of the ICD code
    url = f"https://icd.who.int/browse10/2019/en/JsonGetParentConceptIDsToRoot?ConceptId={code}"
    response = requests.get(url, verify=False)

    new_code = code

    # Check if the response is valid and not null or empty
    if response.status_code == 200:
        data = response.json()
        i = len(code) - 1
        while not data:
            new_code = code[:i]
            url = f"https://icd.who.int/browse10/2019/en/JsonGetParentConceptIDsToRoot?ConceptId={new_code}"
            response = requests.get(url, verify=False)
            data = response.json()
            i -= 1
            if i == 0:
                break

    return new_code, code


def generate_icd_html_links(codes: list[tuple[str, str]]) -> str:
    return f"""
    <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #1f2937;">
        <strong>ICD-10 Codes:</strong>
        <ul>
            {"".join([f'<li><a href="https://icd.who.int/browse10/2019/en#/{code_link}" target="_blank">{code_name}</a></li>' for code_link, code_name in codes])}
        </ul>
    </div>
    """


def format_top_k_icd_codes(text: str, k=5) -> str:
    """Format the top k ICD-10 codes from the given text into HTML."""

    # Extract the ICD-10 codes from the response
    # Make sure its only five codes
    icd_codes = text.split(", ")[:k]

    # Make sure the ICD-10 codes are valid links
    valid_codes = [check_icd_link(c) for c in icd_codes]

    return generate_icd_html_links(valid_codes)
