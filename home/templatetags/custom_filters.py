from django import template
import base64

register = template.Library()

@register.filter
def base64_to_image(value):
    try:
        decoded_data = base64.b64decode(value)
        return decoded_data
    except Exception as e:
        return None
