evaluator_category_prompt = '''
### Instruction ###
You are a professional visual aesthetics evaluator.  
Your task is to assess how well an image represents a specific **Visual Attribute**, based on the provided **Attribute Description**, using the criteria below.

### Evaluation Criteria ###
Rate the visual representation using the following scale:

- **5 – Perfect Match**: The attribute is depicted exactly as described, with all key details visually realized.  
- **4 – Strong Match**: The attribute mostly matches with minor inconsistencies.  
- **3 – Partial Match**: The attribute is somewhat represented, but several important aspects are missing or incorrect.  
- **2 – Weak Match**: Very few elements match the description, and most are inaccurate.  
- **1 – No Match**: The attribute is not represented in the image at all.

### Input ###
- **Attribute Name**: {attribute_name}  
- **Attribute Description**:  
{attribute_description}

> Focus your evaluation only on this specific attribute. Ignore unrelated elements. Be clear and objective in your assessment.

### Output Format ###
Return exactly the following two fields:

**Evaluation Summary**: {{A brief explanation of what matches or does not match.}}  
**Score**: {{X}}
'''
