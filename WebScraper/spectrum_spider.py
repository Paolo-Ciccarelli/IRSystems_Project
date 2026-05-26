# python -m pip install scrapy
# python -m pip install pymupdf

import os
import scrapy
import fitz
import json
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from crawler_indexer import update_index

class SpectrumSpider(scrapy.Spider):
    name = 'spectrum'
    allowed_domains = ["spectrum.library.concordia.ca"]
    start_urls = ["https://spectrum.library.concordia.ca/"]

    def __init__(self, max_files=125, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.max_files = max_files
        self.files_downloaded = 0

        # Directory for temporary PDF files
        self.pdf_dir = "pdfs_extracted"
        if not os.path.exists(self.pdf_dir):
            os.makedirs(self.pdf_dir)
           
        # Directory for all post-crawling output
        self.output_dir = "output"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            
        # The Inverted Index variable
        # Hash Table (key: the unique term, value: its postings list)
        self.index = {}
        
        # Clustering Foundation
        self.docs_metadata = [] # list of dictionaries: {title, degree, year, pdf_url, spectrum_url}
        self.docs_tokens = []   # list of term lists per document, later useful for tf-idf

    # Begin pipeline from the Spectrum home page
    def start_requests(self):
        yield scrapy.Request(
            "https://spectrum.library.concordia.ca/", callback=self.parse_homepage,
        )
    
    # From the homepage, navigate to "Browse" then "By Document Type"
    def parse_homepage(self, response):
        doctype_URL = response.xpath("//a[normalize-space()='by Document Type']/@href").get()
        # Exception handling in the event the crawler cannot find the URL
        if not doctype_URL:
            self.logger.error("ERROR: could not find 'by Document Type' link on home page.")
            return
        print("DEBUG: found 'by Document Type' link", doctype_URL)
        yield response.follow(doctype_URL, callback=self.parse_document_types)
    
    # Stage II: From the Browse By Document Type page, navigate to the 'Thesis' page
    # response: the HTML of the page in question
    def parse_document_types(self, response):
        thesis_url = response.xpath("//a[normalize-space()='Thesis']/@href").get()
        # Exception handling in the event the crawler cannot find the URL
        if not thesis_url:
            self.logger.error("ERROR: could not find 'Thesis' link under the 'Browse by Document Type' page.")
            return
        print("DEBUG: found the 'Thesis' link", thesis_url)
        yield response.follow(thesis_url, callback=self.parse_thesis)
        
    # Stage III: From the Browse By Document Type page, navigate to the 'Thesis' page
    # response: the HTML of the page in question
    def parse_thesis(self, response):
        # Loops over both the "Masters" and "PhD" subdirectories
        for degree_label in ['Masters', 'PhD']:
            degree_url = response.xpath(f"//a[normalize-space()='{degree_label}']/@href").get()
            if degree_url:
                print(f"DEBUG: found the '{degree_label}' link", degree_url)
                yield response.follow(degree_url, callback=self.parse_thesis_years, cb_kwargs={"degree": degree_label})
    
    # Stage IV: From the Masters or PhD page, navigate to the page of a given year 
    # response: the HTML of the page in question
    def parse_thesis_years(self, response, degree):
        year_links = response.xpath("//a[string-length(normalize-space())=4 and number(normalize-space())>=1900 and number(normalize-space())<=2030]/@href").getall()
        for href in year_links:
            year = href.strip("/").split("/")[-1]   # crude but works on Spectrum
            print(f"DEBUG: found the 'Year' link", href)
            yield response.follow(href, callback=self.parse_article_list, cb_kwargs={"degree": degree, "year": year})
    
    # Stage V: From the Year page, navigate to the page of a given PDF article 
    # response: the HTML of the page in question
    def parse_article_list(self, response, degree, year):
        if self.files_downloaded >= self.max_files:
            self.logger.info("ERROR: max files reached, terminating further processing of article list.")
            raise scrapy.exceptions.CloseSpider('max_files_reached')
        
        article_links = response.xpath("//a[contains(@href, '/id/eprint/')]/@href").getall()
        for href in article_links:
            if self.files_downloaded >= self.max_files:
                raise scrapy.exceptions.CloseSpider('max_files_reached')
            #print(f"DEBUG: found the 'Article' link", href)
            yield response.follow(href, callback=self.parse_article, cb_kwargs={"degree": degree, "year": year})
   
    # Stage VI: From the PDF article thesis page, finds
    def parse_article(self, response, degree, year):
        if self.files_downloaded >= self.max_files:
            raise scrapy.exceptions.CloseSpider('max_files_reached')

        # Extracts the title of the given article
        title = response.xpath("//h1/text()").get()
       
        # Finds the link from which to download the PDF
        pdf_link = response.xpath("//a[normalize-space()='Text']/@href").get()
        if not pdf_link:
           self.logger.warning(f"ERROR: no PDF download link found at {response.url}")
           return
        # Expands the link fully, important for later processing
        pdf_complete_url = response.urljoin(pdf_link)
       
        # Ensures the file limit has not been exceeded before proceeding
        if self.files_downloaded >= self.max_files:
           self.logger.info("ERROR: reached max_files limit, terminating PDF request")
           return
       
        yield scrapy.Request(
           pdf_complete_url,
           callback=self.parse_pdf,
           cb_kwargs={
            "degree": degree,
            "year": year,
            "title": title,
            "pdf_url": pdf_complete_url,
            "spectrum_article_url": response.url
        },) 
    
    def extract_pdf_text(self, pdf_path):
        text = ""
        pdf_document = fitz.open(pdf_path)
        for page in pdf_document:
            text += page.get_text()
        pdf_document.close()
        return text
    
    # Downloads PDF temporarily, extracts its text, updates inverted index, and records
    # per-document metadata/tokens.
    def parse_pdf(self, response, degree, year, title, pdf_url, spectrum_article_url):    
        if self.files_downloaded >= self.max_files:
            raise scrapy.exceptions.CloseSpider('max_files_reached')
        temp_pdf_path = os.path.join(self.pdf_dir, f"temp_{self.files_downloaded}.pdf")
        try: 
            with open(temp_pdf_path, 'wb') as f:
                f.write(response.body)
            # Extract the raw text from the given PDF
            text = self.extract_pdf_text(temp_pdf_path)
            
            if text:
                # Assign a new doc_id based on position in docs_meta/docs_token
                doc_id = len(self.docs_metadata)
                
                # Update inverted index and keep tokens for clustering
                terms = update_index(self.index, doc_id, text)
                
                # Store metadata for the given PDF
                self.docs_metadata.append({
                    "doc_id": doc_id,
                    "title": title or "",
                    "degree": degree,
                    "year": year,
                    "pdf_url": pdf_url,
                    "spectrum_url": spectrum_article_url,
                })
                
                self.docs_tokens.append(terms)
                # Increment upper bound counter
                self.files_downloaded += 1
                print(f"DEBUG: parsed document {title} from {spectrum_article_url}")
                #print(f"DEBUG: Extracted {len(text)} characters from {title}")
                #print(f"DEBUG: First 1000 characters: '{text[:1000]}'")
            else:
                self.logger.warning(f"ERROR: no text extracted from {pdf_url}")
        finally:
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
    
    # Executes when the crawling stage terminates
    # Saves all relevant post-crawling output to the directories defined above
    def close(self, response):
        index_path = os.path.join(self.output_dir, "inverted_index.json")
        meta_path = os.path.join(self.output_dir, "pdf_metadata.json")
        tokens_path = os.path.join(self.output_dir, "document_tokens.json")
        
        # Populates 'inverted_index.json' with self.index
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(self.index, f, indent=4, sort_keys=True)
        
        # Populates 'pdf_metadata.json' with self.docs_metadata
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.docs_metadata, f, indent=4, sort_keys=True)
        
        # Populates 'document_tokens.json' with self.docs_tokens
        with open(tokens_path, "w", encoding="utf-8") as f:
            json.dump(self.docs_tokens, f, indent=4, sort_keys=True)    
        
        self.logger.info(f"DEBUG: Inverted Index built from {len(self.docs_metadata)} documents and saved to {index_path}")
        self.logger.info(f"DEBUG: Saved document metadata to {meta_path} and tokens per document to {tokens_path}")
        

process = CrawlerProcess(settings={
    "LOG_LEVEL": "INFO",
    "ROBOTSTXT_OBEY": True,
    #"CONCURRENT_REQUESTS": 1
})
process.crawl(SpectrumSpider, max_files=125)
process.start()