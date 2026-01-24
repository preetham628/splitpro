import os
from typing import Dict, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class BillSplitSuggestion(BaseModel):
    """Structured output for bill split suggestions"""
    veg_amount: float = Field(description="Amount for vegetarian items")
    non_veg_amount: float = Field(description="Amount for non-vegetarian items")
    reasoning: str = Field(description="Brief explanation of the split")
    confidence: str = Field(description="high, medium, or low confidence in the split")


class BillAnalyzer:
    """AI agent to analyze bills and suggest veg/non-veg splits"""
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.3):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key
        )
        self.parser = PydanticOutputParser(pydantic_object=BillSplitSuggestion)
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that analyzes restaurant bills and expenses 
            to suggest how they should be split between vegetarian and non-vegetarian items.

            Analyze the bill or expense description provided and suggest:
            1. How much should be allocated to vegetarian items
            2. How much should be allocated to non-vegetarian items
            3. Your reasoning for the split
            4. Your confidence level (high/medium/low)

            If the bill is unclear or doesn't contain food items, make reasonable assumptions 
            or suggest an even split if no information is available.

            {format_instructions}"""),
            ("human", "Analyze this bill/expense: {bill_text}")
        ])
    
    def analyze_bill(self, bill_text: str) -> Dict[str, any]:
        """
        Analyze a bill and return split suggestions
        
        Returns:
            {
                "veg_amount": float,
                "non_veg_amount": float,
                "reasoning": str,
                "confidence": str
            }
        """
        try:
            prompt = self.prompt_template.format_messages(
                bill_text=bill_text,
                format_instructions=self.parser.get_format_instructions()
            )
            
            response = self.llm.invoke(prompt)
            result = self.parser.parse(response.content)
            
            return {
                "veg_amount": result.veg_amount,
                "non_veg_amount": result.non_veg_amount,
                "reasoning": result.reasoning,
                "confidence": result.confidence
            }
        except Exception as e:
            # Fallback to even split if analysis fails
            print(f"Warning: Bill analysis failed ({e}). Using even split.")
            total = self._extract_amount(bill_text)
            return {
                "veg_amount": total / 2,
                "non_veg_amount": total / 2,
                "reasoning": "Even split (analysis failed)",
                "confidence": "low"
            }
    
    def _extract_amount(self, text: str) -> float:
        """Extract numeric amount from text"""
        import re
        # Look for currency patterns like $120, 120 dollars, etc.
        patterns = [
            r'\$(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*dollars?',
            r'(\d+\.?\d*)\s*USD',
            r'(\d+\.?\d*)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    return float(matches[0])
                except ValueError:
                    continue
        
        return 0.0
