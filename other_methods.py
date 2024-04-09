class methods:
    def parse_response(response):
        # Transfer response byte array to ASCII string
        response_string = ''.join(chr(byte) for byte in response if byte != 0)
        
        # If response include "CRC ERROR", return all response
        if "CRC ERROR" in response_string:
            return response_string
        
        # 'If response is not include "CRC ERROR", extract only important message
        parts = response_string.split(':')
        if len(parts) > 3:
            extracted_response = parts[3]
            return extracted_response
        else:
            return response_string