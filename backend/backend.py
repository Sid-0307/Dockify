from fastapi import FastAPI, HTTPException, Query,File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import re
import requests
import shutil
import zipfile
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DocumentationGenerator:
    def __init__(self):
        try:
            self.model_name = "Salesforce/codet5-base-multi-sum"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            logger.info("Documentation generator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
        
    def extract_zip_repository(self, zip_file_path: str) -> str:
        try:
            extract_path = os.path.join('content', 'temp_repo')
            os.makedirs(extract_path, exist_ok=True)

            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                root_dirs = set()
                for member in zip_ref.namelist():
                    parts = member.split('/')
                    if len(parts) > 1:
                        root_dirs.add(parts[0])
                
                if len(root_dirs) == 1:
                    root_dir = list(root_dirs)[0]
                    extract_path = os.path.join(extract_path, root_dir)
                
                zip_ref.extractall(extract_path)

            logger.info(f"Repository extracted to {extract_path}")
            return extract_path

        except Exception as e:
            logger.error(f"ZIP extraction failed: {e}")
            raise

    def fetch_files_from_local(self, repo_path: str) -> Dict[str, str]:
        files = {}
    
        def walk_directory(current_path):
            app_path = os.path.join(current_path, 'app')
            src_main_path = os.path.join(current_path, 'src', 'main', 'java')
        
            search_paths = []
            if os.path.exists(app_path):
                search_paths.append(app_path)
            if os.path.exists(src_main_path):
                search_paths.append(src_main_path)
        
            if not search_paths:
                search_paths = [current_path]
        
            for search_path in search_paths:
                for root, _, filenames in os.walk(search_path):
                    for filename in filenames:
                        if filename.endswith(('.ts', '.java', '.component.ts',
                                            '.service.ts', '.controller.java',
                                            '.module.ts', '.resolver.ts')):
                            full_path = os.path.join(root, filename)
                            relative_path = os.path.relpath(full_path, current_path)
                        
                            try:
                                with open(full_path, 'r', encoding='utf-8') as f:
                                    files[relative_path] = f.read()
                            except Exception as e:
                                logger.error(f"Error reading file {full_path}: {e}")
    
        walk_directory(repo_path)
    
        if not files:
            logger.warning("No relevant files found in the repository")
    
        return files
    
    def cleanup_temp_files(file_path: str, extracted_path: str):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            if os.path.exists(extracted_path):
                shutil.rmtree(os.path.dirname(extracted_path), ignore_errors=True)
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    def extract_function_description(self, function_code: str) -> str:
        try:
            inputs = self.tokenizer.encode(function_code, return_tensors="pt", max_length=512, truncation=True)
            outputs = self.model.generate(
                inputs,
                max_length=1000,          
                num_beams=5,              
                temperature=0.7,          
                early_stopping=True       
            )
            explanation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return explanation or "Performs an operation"
        except Exception as e:
            logger.warning(f"Function description extraction failed: {e}")
            return "Performs an operation"

    def fetch_files(self, repo_url: str) -> Dict[str, str]:
        try:
            if "github" in repo_url.lower():
                return self._fetch_github_files(repo_url)
            elif "bitbucket" in repo_url.lower():
                return self._fetch_bitbucket_files(repo_url)
            else:
                raise ValueError("Unsupported repository URL")
        except Exception as e:
            logger.error(f"File fetching failed: {e}")
            return {}

    def _fetch_bitbucket_files(self, bitbucket_url: str, branch: str = 'main') -> Dict[str, str]:
        try:
            if not bitbucket_url.startswith('https://'):
                bitbucket_url = f'https://{bitbucket_url}'
            
            parts = bitbucket_url.split('/')
            owner = parts[-4] 
            repo = parts[-3] 
            base_api_url = f"https://api.bitbucket.org/2.0/repositories/{owner}/{repo}/src/{branch}/"
            
            files = {}

            def fetch_recursive(path: str = '') -> None:
                try:
                    response = requests.get(f"{base_api_url}{path}")
                    response.raise_for_status() 
                    contents = response.json()
                    for item in contents['values']:
                        if item['type'] == 'commit_file':
                            # Support multiple file types for Angular and Spring Boot
                            if item['path'].endswith(('.ts', '.java', '.component.ts', '.service.ts', '.controller.java')):
                                file_response = requests.get(item['links']['self']['href'])
                                file_response.raise_for_status()
                                files[item['path']] = file_response.text
                        
                        elif item['type'] == 'commit_directory':
                            fetch_recursive(f"{item['path']}/")

                    if 'next' in contents:
                        next_url = contents['next']
                        response = requests.get(next_url)
                        response.raise_for_status()
                        fetch_recursive(response.json()['values'])

                except Exception as e:
                    logger.error(f"Error fetching files at {path}: {e}")

            fetch_recursive()
            
            if not files:
                logger.warning("No relevant files found in the repository")

            return files

        except Exception as e:
            logger.error(f"Error in fetch_bitbucket_files: {e}")
            return {}

    def _fetch_github_files(self, github_url: str) -> Dict[str, str]:
        try:
            if not github_url.startswith('https://'):
                github_url = f'https://{github_url}'
            
            parts = github_url.split('/')
            owner = parts[-2]
            repo = parts[-1].replace('.git', '')
            
            base_api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
            print(base_api_url)
            
            files = {}
            
            def fetch_recursive(path: str = '') -> None:
                try:
                    response = requests.get(f"{base_api_url}{path}")
                    print("Response",response)
                    contents = response.json()
                    
                    for item in contents:
                        if item['type'] == 'file':
                            if (('src/main' in item['path'] or 'app/' in item['path']) and item['path'].endswith(('.ts', '.java'))):
                                file_response = requests.get(item['download_url'])
                                files[item['path']] = file_response.text
                        
                        elif item['type'] == 'dir':
                            fetch_recursive(f"/{item['path']}")
                
                except Exception as e:
                    logger.error(f"Error fetching files in {path}: {e}")
            
            fetch_recursive()
            return files
        
        except Exception as e:
            logger.error(f"Error in fetch_github_files: {e}")
            return {}
        
    def extract_functions(self, code: str, language: str) -> List[str]:
        if language.lower() in ['typescript', 'ts']:
            function_patterns = [
                r'((?:async\s+)?(?:public|private|protected)?\s*\w+\s*\([^)]*\)\s*:\s*\w+\s*{[^}]*})',
                r'((?:async\s+)?(?:public|private|protected)?\s*\w+\s*\([^)]*\)\s*{[^}]*})',
                r'((?:export\s+)?function\s+\w+\s*\([^)]*\)\s*:\s*\w+\s*{[^}]*})',
                r'((?:export\s+)?function\s+\w+\s*\([^)]*\)\s*{[^}]*})'
            ]
        elif language.lower() in ['java']:
            function_patterns = [
                r'((?:public|private|protected)\s+(?:static\s+)?(?:\w+\s+)?\w+\s*\([^)]*\)\s*{[^}]*})',
                r'((?:@\w+\s+)*(?:public|private|protected)\s+(?:static\s+)?(?:\w+\s+)?\w+\s*\([^)]*\)\s*{[^}]*})'
            ]
        else:
            return []
        
        functions = []
        for pattern in function_patterns:
            matches = re.findall(pattern, code, re.DOTALL)
            functions.extend(matches)
        
        return functions

    async def generate_function_documentation(self, function: str, language: str) -> str:
        try:
            if language.lower() in ['typescript', 'ts']:
                name_match = re.search(r'(?:function\s+|(?:public|private|protected)?\s*)(\w+)\s*\(', function)
                params_match = re.search(r'\(([^)]*)\)', function)
                return_type_match = re.search(r':\s*(\w+)\s*{', function)
            elif language.lower() in ['java']:
                name_match = re.search(r'(?:(?:public|private|protected)\s+(?:static\s+)?(?:\w+\s+)?)?(\w+)\s*\(', function)
                params_match = re.search(r'\(([^)]*)\)', function)
                return_type_match = re.search(r'(?:(?:public|private|protected)\s+(?:static\s+)?)([\w<>]+)\s+\w+\s*\(', function)
            else:
                return "Unsupported language"

            function_name = name_match.group(1) if name_match else "UnknownFunction"
            params = params_match.group(1) if params_match else "None"
            return_type = return_type_match.group(1) if return_type_match else "void"

            description = self.extract_function_description(function)
            
            param_details = self._parse_parameters(params, language)
            
            doc = f"### Method: {function_name}()\n"
            
            doc += f"- **Description**: {description}\n"
            doc += f"- **Return Type**: {return_type}\n"
            doc += f"- **Parameters**: {param_details}\n"
            doc += f"- **Annotations**: Component\n"
            
            return doc

        except Exception as e:
            logger.error(f"Function documentation generation failed: {e}")
            return f"### Method: {function_name}\n- Unable to generate documentation: {e}\n"

    def _parse_parameters(self, params: str, language: str) -> str:
        if not params.strip() or params.strip().lower() == 'none':
            return "None"

        param_docs = []
        param_parts = [p.strip() for p in params.split(',')]
        
        for param in param_parts:
            if language.lower() in ['typescript', 'ts']:
                match = re.match(r'(\w+)\s*:\s*(\w+)', param)
            elif language.lower() in ['java']:
                match = re.match(r'(\w+(?:<\w+>)?)\s+(\w+)', param)
            else:
                return "None"

            if match:
                param_type, param_name = match.groups()
                param_docs.append(f"{param_name}: {param_type}")
            else:
                param_docs.append(param)

        return ", ".join(param_docs) if param_docs else "None"

    async def generate_comprehensive_documentation(self, source: str) -> str:
        try:
            if source.startswith(('http://', 'https://')):
                files = self.fetch_files(source)
            else:
                files = self.fetch_files_from_local(source)
            
            if not files:
                logger.warning("No files found in the repository")
                return "No files found in the repository."
            
            comprehensive_doc = ""
            
            for filepath, code in files.items():
                language = 'ts' if filepath.endswith(('.ts', '.component.ts', '.service.ts')) else 'java'
                
                functions = self.extract_functions(code, language)
                
                if not functions:
                    continue 
                
                comprehensive_doc += f"### File: {filepath}\n"
                
                for function in functions:
                    try:
                        func_doc = await self.generate_function_documentation(function, language)
                        comprehensive_doc += func_doc + "\n"
                    
                    except Exception as e:
                        logger.error(f"Error processing function: {e}")
                        comprehensive_doc += f"### Method: Error\n- Unable to generate documentation: {e}\n\n"
            
            if not comprehensive_doc:
                return "No functions found in the repository."
            
            return comprehensive_doc

        except Exception as e:
            logger.error(f"Comprehensive documentation generation failed: {e}")
            return f"Documentation generation failed: {e}"

    async def save_documentation(self, documentation: str, output_dir: str = '/content/docs') -> str:
        try:
            # os.makedirs(output_dir, exist_ok=True)s
            
            doc_filename = os.path.join('content', 'documentation.md')
            
            with open(doc_filename, 'w', encoding='utf-8') as f:
                f.write(documentation)
            
            logger.info(f"Documentation saved to {doc_filename}")
            return doc_filename

        except Exception as e:
            logger.error(f"Documentation saving failed: {e}")
            raise

@app.get("/generate-docs")
async def generate_docs(repo_url: str = Query(..., description="Repository URL")):
    if not repo_url:
        raise HTTPException(status_code=400, detail="Repository URL is required")

    try:
        doc_generator = DocumentationGenerator()
        comprehensive_doc = await doc_generator.generate_comprehensive_documentation(repo_url)
        
        doc_file = await doc_generator.save_documentation(comprehensive_doc)
        
        return FileResponse(
            doc_file,
            media_type="text/markdown",
            filename="documentation.md",
        )
    
    except Exception as e:
        logger.error(f"Documentation generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Documentation generation failed: {str(e)}")
    
@app.post("/upload-repo")
async def upload_repository(file: UploadFile = File(...)):
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only ZIP files are supported")

    try:
        upload_dir = '/content/uploads'
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        
        with open(file_path, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        doc_generator = DocumentationGenerator()
        
        repo_path = doc_generator.extract_zip_repository(file_path)
        
        comprehensive_doc = await doc_generator.generate_comprehensive_documentation(repo_path)
        
        doc_file = await doc_generator.save_documentation(comprehensive_doc)
        
        # doc_generator.cleanup_temp_files(file_path, repo_path)
        
        return FileResponse(
            doc_file,
            media_type="text/markdown",
            filename="documentation.md",
        )
    
    except Exception as e:
        logger.error(f"Repository documentation generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Documentation generation failed: {str(e)}")


os.makedirs('/content/uploads', exist_ok=True)
os.makedirs('/content/docs', exist_ok=True)
os.makedirs('/content/temp_repo', exist_ok=True)