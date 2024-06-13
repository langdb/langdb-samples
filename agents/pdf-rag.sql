-- Create a table to store content from PDF files
CREATE TABLE pdfs (
  id UUID DEFAULT generateUUIDv4(),
  content `String`, 
  metadata `String`,
   
) 
engine = MergeTree
order by (id, content);

-- Load a PDF file 
INSERT INTO pdfs(content, metadata)
SELECT content, metadata from 
load_pdf_ ('file:///var/lib/clickhouse/user_files/apple.pdf');


CREATE TABLE pdf_embeddings (
  id UUID,
  embeddings `Array`(`Float32`)
) 
engine = MergeTree
order by id;

-- Maintain records that dont have embeddings yet
SPAWN TASK generate_embeddings
    BEGIN
INSERT INTO pdf_embeddings
select p.id, embed(content) from pdfs as p LEFT JOIN pdf_embeddings as pe on p.id = pe.id
where p.id != pe.id
order by p.id
limit 10
END EVERY 5 second;


-- CREATE AGENT to find similar documents based on embeddings


CREATE AGENT similar(query String "Query to search similar sections in pdf documents") AS
WITH tbl AS (
  SELECT CAST(embed($query) AS `Array`(`Float32`)) AS query
)
SELECT 
  p.id as id, 
  content, 
  cosineDistance(embeddings, query) AS similarity 
FROM 
  pdf_embeddings AS pe 
JOIN 
  pdfs AS p ON p.id = pe.id
CROSS JOIN 
  tbl 
ORDER BY 
  similarity ASC 
  LIMIT 5

-- Now you can query similar documents like this:  
select * from similar('PLEASE NOTE: THERE IS NO PROOF OF CLAIM FORM FOR')



CREATE MODEL doc_search(
  input
) ENGINE = OpenAI(api_key = 'sk-proj-xxx', model_name = 'gpt-3.5-turbo')
PROMPT (
  system "Use the tool 'similar' to query for similar documents based on user query and respond with relevant docuemnts.",
  human "{{input}}")
TOOLS (similar)
SETTINGS retries = 1;

select * from doc_search('THERE IS NO PROOF OF CLAIM FORM FOR')