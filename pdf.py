import os
import logging
from typing import List, Optional, Dict, Union, Tuple
import asyncio
from enum import Enum
from pathlib import Path
import json
import requests
from pdf2image import convert_from_path 
import pytesseract
from PIL import Image
import cv2
import numpy as np
import fitz  # PyMuPDF
import re
from dataclasses import dataclass
from abc import ABC, abstractmethod
import pandas as pd
import tabula
from table2ascii import table2ascii
import camelot

# Configuration avancée
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFType(Enum):
    SEARCHABLE = "searchable"
    SCANNED = "scanned"
    MIXED = "mixed"
    IMAGE_ONLY = "image_only"
    TABLE_HEAVY = "table_heavy"

class ContentType(Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    MIXED = "mixed"

@dataclass
class TableData:
    headers: List[str]
    rows: List[List[str]]
    page_number: int
    position: Dict[str, float]  # x1, y1, x2, y2
    confidence_score: float

@dataclass
class PDFMetadata:
    file_path: Path
    pdf_type: PDFType
    page_count: int
    has_images: bool
    has_text: bool
    has_tables: bool
    file_size: int
    creation_date: str
    content_types: List[ContentType]
    table_count: int
    image_count: int

class PDFProcessingConfig:
    def __init__(self):
        self.OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
        self.OPENAI_MODEL: str = "gpt-4o"
        self.OCR_LANGUAGE: str = "eng"
        self.MAX_TOKENS: int = 2048
        self.TEMPERATURE: float = 0.3
        self.OUTPUT_FORMAT: str = "markdown"
        self.TABLE_EXTRACTION_METHODS = ["camelot", "tabula", "ocr"]
        self.MIN_TABLE_CONFIDENCE = 0.7
        self.ENABLE_TABLE_RESTRUCTURING = True
        self.TABLE_FORMAT_STYLE = "markdown"  # or "ascii"
        self.IMAGE_DPI = 300
        self.OCR_CONFIG = {
            "lang": "eng",
            "config": "--oem 3 --psm 6"
        }

class OpenAIProcessor:
    def __init__(self, config: PDFProcessingConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {config.OPENAI_API_KEY}",
            "Content-Type": "application/json"
        })

    def process_text(self, text: str, instruction: str = None) -> Optional[str]:
        try:
            if not instruction:
                instruction = "Format and correct the following text into well-structured markdown."

            data = {
                "model": self.config.OPENAI_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": instruction}]
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": text}]
                    }
                ],
                "response_format": {"type": "text"},
                "temperature": self.config.TEMPERATURE,
                "max_tokens": self.config.MAX_TOKENS
            }

            response = self.session.post(
                "https://api.openai.com/v1/chat/completions",
                json=data
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']

        except Exception as e:
            logger.error(f"Error in OpenAI processing: {e}")
            return None

    def restructure_table(self, table_data: TableData) -> str:
        try:
            # Convert table data to string representation
            table_str = "Headers: " + ", ".join(table_data.headers) + "\n"
            for row in table_data.rows:
                table_str += "Row: " + ", ".join(row) + "\n"

            instruction = """
            Restructure this table data into a well-formatted markdown structure.
            For each row, create a section with:
            1. A header for the main item
            2. Bullet points for other columns
            3. Ensure proper formatting and capitalization
            Example format:
            # [First Column Value]
            - [Second Column Header]: [Second Column Value]
            - [Third Column Header]: [Third Column Value]
            """

            return self.process_text(table_str, instruction)

        except Exception as e:
            logger.error(f"Error in table restructuring: {e}")
            return None

class TableExtractor:
    def __init__(self, config: PDFProcessingConfig):
        self.config = config

    def extract_tables_camelot(self, pdf_path: Path) -> List[TableData]:
        tables = camelot.read_pdf(str(pdf_path), pages='all')
        return [self._convert_to_table_data(table, idx) for idx, table in enumerate(tables)]

    def extract_tables_tabula(self, pdf_path: Path) -> List[TableData]:
        tables = tabula.read_pdf(str(pdf_path), pages='all')
        return [self._convert_to_table_data(table, idx) for idx, table in enumerate(tables)]

    def extract_tables_ocr(self, image: Image.Image) -> List[TableData]:
        try:
            # Prétraitement de l'image
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            # Détection des lignes horizontales et verticales
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
            horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

            # Combinaison des lignes
            table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
            
            # Trouver les contours
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            tables = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 100 and h > 100:  # Filtrer les petits contours
                    table_region = image.crop((x, y, x+w, y+h))
                    text = pytesseract.image_to_string(table_region, config=self.config.OCR_CONFIG['config'])
                    
                    # Analyse basique du texte pour extraire les données du tableau
                    rows = text.split('\n')
                    table_data = []
                    for row in rows:
                        if row.strip():
                            cells = row.split()
                            if cells:
                                table_data.append(cells)
                    
                    if len(table_data) > 1:  # Au moins une en-tête et une ligne
                        tables.append(TableData(
                            headers=table_data[0],
                            rows=table_data[1:],
                            page_number=0,  # À ajuster selon le contexte
                            position={"x1": x, "y1": y, "x2": x+w, "y2": y+h},
                            confidence_score=0.8  # Score arbitraire, à affiner
                        ))
            
            return tables
        except Exception as e:
            logger.error(f"Error in OCR table extraction: {e}")
            return []

    def _convert_to_table_data(self, table, page_number: int) -> TableData:
        if isinstance(table, pd.DataFrame):
            headers = table.columns.tolist()
            rows = table.values.tolist()
        else:
            # Handle camelot table format
            headers = table.df.iloc[0].tolist()
            rows = table.df.iloc[1:].values.tolist()

        # Nettoyage des données
        headers = [str(h).strip() for h in headers]
        rows = [[str(cell).strip() for cell in row] for row in rows]

        # Calcul du score de confiance basé sur la qualité des données
        confidence_score = self._calculate_confidence_score(headers, rows)

        return TableData(
            headers=headers,
            rows=rows,
            page_number=page_number,
            position={"x1": 0, "y1": 0, "x2": 0, "y2": 0},
            confidence_score=confidence_score
        )

    def _calculate_confidence_score(self, headers: List[str], rows: List[List[str]]) -> float:
        try:
            # Vérification de la cohérence des données
            if not headers or not rows:
                return 0.0

            # Vérification de la longueur des lignes
            expected_length = len(headers)
            length_consistency = sum(1 for row in rows if len(row) == expected_length) / len(rows)

            # Vérification de la qualité des données
            empty_cells = sum(1 for row in rows for cell in row if not cell.strip())
            total_cells = len(rows) * len(headers)
            data_quality = 1 - (empty_cells / total_cells if total_cells > 0 else 0)

            # Score final
            return min(1.0, (length_consistency * 0.6 + data_quality * 0.4))

        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 0.0
class PDFProcessor(ABC):
    def __init__(self, config: PDFProcessingConfig):
        self.config = config
        self.openai_processor = OpenAIProcessor(config)
        self.table_extractor = TableExtractor(config)

    @abstractmethod
    async def process(self, pdf_path: Path) -> Dict[str, Union[str, List[TableData]]]:
        pass

    async def process_tables(self, pdf_path: Path) -> List[TableData]:
        tables = []
        for method in self.config.TABLE_EXTRACTION_METHODS:
            try:
                if method == "camelot":
                    extracted_tables = self.table_extractor.extract_tables_camelot(pdf_path)
                    tables.extend(extracted_tables)
                elif method == "tabula":
                    extracted_tables = self.table_extractor.extract_tables_tabula(pdf_path)
                    tables.extend(extracted_tables)
                elif method == "ocr":
                    # Pour OCR, nous devons d'abord convertir les pages en images
                    images = convert_from_path(str(pdf_path))
                    for image in images:
                        extracted_tables = self.table_extractor.extract_tables_ocr(image)
                        tables.extend(extracted_tables)
            except Exception as e:
                logger.warning(f"Failed to extract tables using {method}: {e}")

        # Filtrer les tables selon le score de confiance
        filtered_tables = [
            table for table in tables 
            if table.confidence_score >= self.config.MIN_TABLE_CONFIDENCE
        ]

        # Déduplication des tables
        unique_tables = self._deduplicate_tables(filtered_tables)

        return unique_tables

    def _deduplicate_tables(self, tables: List[TableData]) -> List[TableData]:
        """Supprime les tables en double basé sur leur contenu."""
        unique_tables = []
        seen_contents = set()

        for table in tables:
            # Créer une représentation hashable du contenu de la table
            content_hash = str(table.headers) + str(table.rows)
            
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_tables.append(table)

        return unique_tables

class SearchablePDFProcessor(PDFProcessor):
    async def process(self, pdf_path: Path) -> Dict[str, Union[str, List[TableData]]]:
        doc = fitz.open(pdf_path)
        text_content = ""
        tables = await self.process_tables(pdf_path)
        
        # Extraction du texte page par page
        for page in doc:
            text_content += page.get_text()

        # Traitement du texte principal
        processed_text = self.openai_processor.process_text(text_content)
        
        if self.config.ENABLE_TABLE_RESTRUCTURING and tables:
            restructured_tables = []
            for table in tables:
                restructured = self.openai_processor.restructure_table(table)
                if restructured:
                    restructured_tables.append(restructured)
            
            # Combine text and restructured tables
            final_content = processed_text + "\n\n" + "\n\n".join(restructured_tables)
        else:
            final_content = processed_text

        return {
            "text": final_content,
            "tables": tables
        }

class ScannedPDFProcessor(PDFProcessor):
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        # Conversion en array numpy
        img_array = np.array(image)
        
        # Conversion en niveaux de gris
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Débruitage
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Amélioration du contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Binarisation adaptative
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return Image.fromarray(binary)

    async def process(self, pdf_path: Path) -> Dict[str, Union[str, List[TableData]]]:
        # Conversion du PDF en images
        images = convert_from_path(str(pdf_path), dpi=self.config.IMAGE_DPI)
        
        extracted_text = ""
        for image in images:
            # Prétraitement de l'image
            preprocessed = self.preprocess_image(image)
            
            # OCR avec configuration personnalisée
            page_text = pytesseract.image_to_string(
                preprocessed,
                lang=self.config.OCR_LANGUAGE,
                config=self.config.OCR_CONFIG['config']
            )
            extracted_text += page_text + "\n\n"

        # Extraction et traitement des tables
        tables = await self.process_tables(pdf_path)
        
        # Traitement du texte avec OpenAI
        processed_text = self.openai_processor.process_text(extracted_text)
        
        # Restructuration des tables si activée
        if self.config.ENABLE_TABLE_RESTRUCTURING and tables:
            restructured_tables = []
            for table in tables:
                restructured = self.openai_processor.restructure_table(table)
                if restructured:
                    restructured_tables.append(restructured)
            
            final_content = processed_text + "\n\n" + "\n\n".join(restructured_tables)
        else:
            final_content = processed_text

        return {
            "text": final_content,
            "tables": tables
        }

class MixedPDFProcessor(PDFProcessor):
    async def process(self, pdf_path: Path) -> Dict[str, Union[str, List[TableData]]]:
        doc = fitz.open(pdf_path)
        text_content = ""
        tables = await self.process_tables(pdf_path)
        
        for page in doc:
            # Essayer d'abord l'extraction directe du texte
            page_text = page.get_text()
            
            # Si peu ou pas de texte trouvé, utiliser l'OCR
            if len(page_text.strip()) < 50:  # Seuil arbitraire
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Prétraitement de l'image
                preprocessed = ScannedPDFProcessor(self.config).preprocess_image(img)
                
                # OCR
                page_text = pytesseract.image_to_string(
                    preprocessed,
                    lang=self.config.OCR_LANGUAGE,
                    config=self.config.OCR_CONFIG['config']
                )
            
            text_content += page_text + "\n\n"

        # Traitement du texte avec OpenAI
        processed_text = self.openai_processor.process_text(text_content)
        
        # Restructuration des tables
        if self.config.ENABLE_TABLE_RESTRUCTURING and tables:
            restructured_tables = []
            for table in tables:
                restructured = self.openai_processor.restructure_table(table)
                if restructured:
                    restructured_tables.append(restructured)
            
            final_content = processed_text + "\n\n" + "\n\n".join(restructured_tables)
        else:
            final_content = processed_text

        return {
            "text": final_content,
            "tables": tables
        }

class TableHeavyPDFProcessor(PDFProcessor):
    async def process(self, pdf_path: Path) -> Dict[str, Union[str, List[TableData]]]:
        # Extraction prioritaire des tables
        tables = await self.process_tables(pdf_path)
        
        # Extraction du texte entre les tables
        doc = fitz.open(pdf_path)
        text_content = ""
        
        for page in doc:
            page_text = page.get_text()
            # TODO: Améliorer la détection des zones de texte entre les tables
            text_content += page_text + "\n\n"

        # Traitement spécifique pour les documents riches en tableaux
        instruction = """
        Process this text with special attention to:
        1. Preserve table references and context
        2. Maintain relationships between tables and explanatory text
        3. Format table captions and references properly
        4. Structure the content to highlight table-related information
        """
        
        processed_text = self.openai_processor.process_text(text_content, instruction)
        
        # Restructuration avancée des tables
        if self.config.ENABLE_TABLE_RESTRUCTURING and tables:
            restructured_tables = []
            for table in tables:
                restructured = self.openai_processor.restructure_table(table)
                if restructured:
                    restructured_tables.append(restructured)
            
            # Intégration intelligente des tables dans le texte
            final_content = self._integrate_tables_with_text(processed_text, restructured_tables)
        else:
            final_content = processed_text

        return {
            "text": final_content,
            "tables": tables
        }

    def _integrate_tables_with_text(self, text: str, restructured_tables: List[str]) -> str:
        """Intègre intelligemment les tables dans le texte."""
        # TODO: Implémenter une logique plus sophistiquée pour l'intégration
        # Pour l'instant, simple concaténation avec des séparateurs
        return text + "\n\n" + "\n\n".join(restructured_tables)

class PDFAnalyzer:
    @staticmethod
    def analyze_pdf(pdf_path: Path) -> PDFMetadata:
        doc = fitz.open(pdf_path)
        has_text = False
        has_images = False
        has_tables = False
        content_types = []
        table_count = 0
        image_count = 0

        # Analyse page par page
        for page in doc:
            # Détection de texte
            page_text = page.get_text()
            if page_text.strip():
                has_text = True
                if ContentType.TEXT not in content_types:
                    content_types.append(ContentType.TEXT)

            # Détection d'images
            images = page.get_images()
            if images:
                has_images = True
                image_count += len(images)
                if ContentType.IMAGE not in content_types:
                    content_types.append(ContentType.IMAGE)

            # Détection de tableaux
            # Plusieurs méthodes de détection
            if (
                re.search(r'\b\w+\s*\|\s*\w+\b', page_text) or  # Motif de tableau avec |
                re.search(r'\b\w+\s*\t\s*\w+\b', page_text) or  # Motif avec tabulations
                re.search(r'\b\w+\s{2,}\w+\b', page_text) or    # Espaces multiples
                len(page.search_for("table")) > 0                # Mot "table"
            ):
                has_tables = True
                table_count += 1
                if ContentType.TABLE not in content_types:
                    content_types.append(ContentType.TABLE)

        # Détermination du type principal de PDF
        if len(content_types) > 1:
            pdf_type = PDFType.MIXED
        elif has_text and not has_images and not has_tables:
            pdf_type = PDFType.SEARCHABLE
        elif has_images and not has_text:
            pdf_type = PDFType.IMAGE_ONLY
        elif has_tables and (table_count / len(doc)) > 0.5:
            pdf_type = PDFType.TABLE_HEAVY
        else:
            pdf_type = PDFType.SCANNED

        return PDFMetadata(
            file_path=pdf_path,
            pdf_type=pdf_type,
            page_count=len(doc),
            has_images=has_images,
            has_text=has_text,
            has_tables=has_tables,
            file_size=os.path.getsize(pdf_path),
            creation_date=doc.metadata.get('creationDate', ''),
            content_types=content_types,
            table_count=table_count,
            image_count=image_count
        )

class PDFPipeline:
    def __init__(self, config: PDFProcessingConfig):
        self.config = config
        self.processors = {
            PDFType.SEARCHABLE: SearchablePDFProcessor(config),
            PDFType.SCANNED: ScannedPDFProcessor(config),
            PDFType.MIXED: MixedPDFProcessor(config),
            PDFType.IMAGE_ONLY: ScannedPDFProcessor(config),
            PDFType.TABLE_HEAVY: TableHeavyPDFProcessor(config)
        }

    async def process_pdf(self, pdf_path: Union[str, Path]) -> Dict:
        pdf_path = Path(pdf_path)
        try:
            # Validation du fichier
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            if pdf_path.suffix.lower() != '.pdf':
                raise ValueError(f"File is not a PDF: {pdf_path}")

            # Analyse du PDF
            logger.info(f"Starting analysis of PDF: {pdf_path}")
            metadata = PDFAnalyzer.analyze_pdf(pdf_path)
            logger.info(f"PDF Type detected: {metadata.pdf_type}")
            logger.info(f"Content types found: {[ct.value for ct in metadata.content_types]}")

            # Sélection et utilisation du processeur approprié
            processor = self.processors[metadata.pdf_type]
            logger.info(f"Using processor: {processor.__class__.__name__}")
            
            # Traitement du PDF
            result = await processor.process(pdf_path)

            # Génération des chemins de sortie
            base_path = pdf_path.with_suffix('')
            output_paths = {
                "markdown": base_path.with_suffix(f'.{self.config.OUTPUT_FORMAT}'),
                "metadata": base_path.with_suffix('.metadata.json'),
                "tables": base_path.with_suffix('.tables.json')
            }

            # Sauvegarde des résultats
            # Texte principal
            with open(output_paths["markdown"], 'w', encoding='utf-8') as f:
                f.write(result["text"])

            # Métadonnées
            metadata_dict = {
                "file_info": {
                    "original_file": str(pdf_path),
                    "file_size": metadata.file_size,
                    "creation_date": metadata.creation_date
                },
                "content_analysis": {
                    "pdf_type": metadata.pdf_type.value,
                    "page_count": metadata.page_count,
                    "has_images": metadata.has_images,
                    "has_text": metadata.has_text,
                    "has_tables": metadata.has_tables,
                    "content_types": [ct.value for ct in metadata.content_types],
                    "table_count": metadata.table_count,
                    "image_count": metadata.image_count
                },
                "processing_info": {
                    "processor_used": processor.__class__.__name__,
                    "processing_date": datetime.now().isoformat(),
                    "config_used": {
                        "model": self.config.OPENAI_MODEL,
                        "table_extraction_methods": self.config.TABLE_EXTRACTION_METHODS,
                        "output_format": self.config.OUTPUT_FORMAT
                    }
                }
            }

            with open(output_paths["metadata"], 'w', encoding='utf-8') as f:
                json.dump(metadata_dict, f, indent=2)

            # Tables extraites
            if result.get("tables"):
                tables_data = [
                    {
                        "headers": table.headers,
                        "rows": table.rows,
                        "page_number": table.page_number,
                        "position": table.position,
                        "confidence_score": table.confidence_score
                    }
                    for table in result["tables"]
                ]
                with open(output_paths["tables"], 'w', encoding='utf-8') as f:
                    json.dump(tables_data, f, indent=2)

            return {
                "status": "success",
                "metadata": metadata_dict,
                "output_paths": {k: str(v) for k, v in output_paths.items()},
                "processed_text_length": len(result["text"]),
                "tables_processed": len(result.get("tables", [])),
                "content_types": [ct.value for ct in metadata.content_types]
            }

        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "metadata": metadata.__dict__ if 'metadata' in locals() else None
            }

async def main():
    # Configuration
    config = PDFProcessingConfig()
    
    # Création du pipeline
    pipeline = PDFPipeline(config)
    
    # Exemple de traitement d'un PDF
    pdf_path = "example.pdf"
    result = await pipeline.process_pdf(pdf_path)
    
    # Affichage des résultats
    if result["status"] == "success":
        logger.info("PDF processed successfully!")
        logger.info(f"Output files:")
        for file_type, path in result["output_paths"].items():
            logger.info(f"- {file_type}: {path}")
        logger.info(f"Content types found: {result['content_types']}")
        logger.info(f"Number of tables processed: {result['tables_processed']}")
    else:
        logger.error(f"Failed to process PDF: {result['error']}")
        logger.error(result['traceback'])

if __name__ == "__main__":
    # Ajout des imports manquants
    from datetime import datetime
    import traceback
    
    # Configuration du logging pour le développement
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Exécution du programme
    asyncio.run(main())


config = PDFProcessingConfig()
pipeline = PDFPipeline(config)
result = await pipeline.process_pdf("votre_document.pdf")
