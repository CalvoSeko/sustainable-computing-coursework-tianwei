import requests

def get_postcode_intensity(postcode):
    # The API only needs the first part of the postcode (outward code)
    outward_code = postcode.split()[0].upper()
    url = f"https://api.carbonintensity.org.uk/regional/postcode/{outward_code}"
    
    headers = {
        'Accept': 'application/json'
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() # Check for HTTP errors
        data = response.json()
        
        # Extracting relevant information
        region_data = data['data'][0]
        postcode_area = region_data['postcode']
        intensity = region_data['data'][0]['intensity']
        
        print(f"Postcode: {postcode_area}")
        print(f"Forecast Intensity: {intensity['forecast']} gCO2/kWh")
        print(f"Index: {intensity['index']}")
        
        return intensity
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")

# Example usage
get_postcode_intensity("OX1")
