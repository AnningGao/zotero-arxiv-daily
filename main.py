import arxiv

def _get_pdf_url_patch(links) -> str:
    """
    Finds the PDF link among a result's links and returns its URL.
    Should only be called once for a given `Result`, in its constructor.
    After construction, the URL should be available in `Result.pdf_url`.
    """
    pdf_urls = [link.href for link in links if "pdf" in link.href]
    if len(pdf_urls) == 0:
        return None
    return pdf_urls[0]

arxiv.Result._get_pdf_url = _get_pdf_url_patch

import re
import argparse
import os
import sys
from dotenv import load_dotenv
load_dotenv(override=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pyzotero import zotero
from recommender import rerank_paper
from construct_email import render_email, render_email_grouped, send_email
from tqdm import trange,tqdm
from loguru import logger
from gitignore_parser import parse_gitignore
from tempfile import mkstemp
from paper import ArxivPaper
from llm import set_global_llm
import feedparser

def get_zotero_corpus(id:str,key:str) -> tuple[list[dict], set[str]]:
    zot = zotero.Zotero(id, 'user', key)
    collections = zot.everything(zot.collections())
    collections = {c['key']:c for c in collections}
    corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
    corpus = [c for c in corpus if c['data']['abstractNote'] != '']
    def get_collection_path(col_key:str) -> str:
        if p := collections[col_key]['data']['parentCollection']:
            return get_collection_path(p) + '/' + collections[col_key]['data']['name']
        else:
            return collections[col_key]['data']['name']
    all_collection_paths = {get_collection_path(k) for k in collections}
    for c in corpus:
        paths = [get_collection_path(col) for col in c['data']['collections']]
        c['paths'] = paths
    return corpus, all_collection_paths

def parse_categories(filepath:str, valid_paths:set[str]) -> list[tuple[str, list[str], int]]:
    """Parse .categories file. Returns list of (label, [collection_paths], max_num).
    Raises ValueError if a path does not exist in the Zotero library.
    """
    groups = []
    with open(filepath) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if ';' not in line:
                raise ValueError(f"Line {lineno}: missing ';' separator: {line!r}")
            lhs, rhs = line.rsplit(';', 1)
            try:
                max_num = int(rhs.strip())
            except ValueError:
                raise ValueError(f"Line {lineno}: max paper count must be an integer, got {rhs.strip()!r}")
            paths = re.findall(r'\(([^)]+)\)', lhs)
            if not paths:
                raise ValueError(f"Line {lineno}: no collection paths found in parentheses")
            paths = [p.strip() for p in paths]
            for path in paths:
                if path not in valid_paths:
                    raise ValueError(f"Line {lineno}: collection path {path!r} does not exist in Zotero library")
            groups.append((lhs.strip(), paths, max_num))
    return groups


def filter_corpus_by_paths(corpus:list[dict], paths:list[str]) -> list[dict]:
    """Return papers whose collection path starts with any of the given paths."""
    result = []
    for paper in corpus:
        for paper_path in paper['paths']:
            if any(paper_path == p or paper_path.startswith(p + '/') for p in paths):
                result.append(paper)
                break
    return result


def filter_corpus(corpus:list[dict], pattern:str) -> list[dict]:
    _,filename = mkstemp()
    with open(filename,'w') as file:
        file.write(pattern)
    matcher = parse_gitignore(filename,base_dir='./')
    new_corpus = []
    for c in corpus:
        match_results = [matcher(p) for p in c['paths']]
        if not any(match_results):
            new_corpus.append(c)
    os.remove(filename)
    return new_corpus


def get_arxiv_paper(query:str, debug:bool=False) -> list[ArxivPaper]:
    client = arxiv.Client(num_retries=10,delay_seconds=10)
    feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
    if 'Feed error for query' in feed.feed.title:
        raise Exception(f"Invalid ARXIV_QUERY: {query}.")
    if not debug:
        papers = []
        all_paper_ids = [i.id.removeprefix("oai:arXiv.org:") for i in feed.entries if i.arxiv_announce_type == 'new']
        bar = tqdm(total=len(all_paper_ids),desc="Retrieving Arxiv papers")
        for i in range(0,len(all_paper_ids),20):
            search = arxiv.Search(id_list=all_paper_ids[i:i+20])
            batch = [ArxivPaper(p) for p in client.results(search)]
            bar.update(len(batch))
            papers.extend(batch)
        bar.close()

    else:
        logger.debug("Retrieve 5 arxiv papers regardless of the date.")
        search = arxiv.Search(query='cat:cs.AI', sort_by=arxiv.SortCriterion.SubmittedDate)
        papers = []
        for i in client.results(search):
            papers.append(ArxivPaper(i))
            if len(papers) == 5:
                break

    return papers



parser = argparse.ArgumentParser(description='Recommender system for academic papers')

def add_argument(*args, **kwargs):
    def get_env(key:str,default=None):
        # handle environment variables generated at Workflow runtime
        # Unset environment variables are passed as '', we should treat them as None
        v = os.environ.get(key)
        if v == '' or v is None:
            return default
        return v
    parser.add_argument(*args, **kwargs)
    arg_full_name = kwargs.get('dest',args[-1][2:])
    env_name = arg_full_name.upper()
    env_value = get_env(env_name)
    if env_value is not None:
        #convert env_value to the specified type
        if kwargs.get('type') == bool:
            env_value = env_value.lower() in ['true','1']
        else:
            env_value = kwargs.get('type')(env_value)
        parser.set_defaults(**{arg_full_name:env_value})


if __name__ == '__main__':
    
    add_argument('--zotero_id', type=str, help='Zotero user ID')
    add_argument('--zotero_key', type=str, help='Zotero API key')
    add_argument('--zotero_ignore',type=str,help='Zotero collection to ignore, using gitignore-style pattern.')
    add_argument('--send_empty', type=bool, help='If get no arxiv paper, send empty email',default=False)
    add_argument('--max_paper_num', type=int, help='Maximum number of papers to recommend',default=100)
    add_argument('--arxiv_query', type=str, help='Arxiv search query')
    add_argument('--smtp_server', type=str, help='SMTP server')
    add_argument('--smtp_port', type=int, help='SMTP port')
    add_argument('--sender', type=str, help='Sender email address')
    add_argument('--receiver', type=str, help='Receiver email address')
    add_argument('--sender_password', type=str, help='Sender email password')
    add_argument(
        "--use_llm_api",
        type=bool,
        help="Use OpenAI API to generate TLDR",
        default=False,
    )
    add_argument(
        "--openai_api_key",
        type=str,
        help="OpenAI API key",
        default=None,
    )
    add_argument(
        "--openai_api_base",
        type=str,
        help="OpenAI API base URL",
        default="https://api.openai.com/v1",
    )
    add_argument(
        "--model_name",
        type=str,
        help="LLM Model Name",
        default="gpt-4o",
    )
    add_argument(
        "--language",
        type=str,
        help="Language of TLDR",
        default="English",
    )
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    assert (
        not args.use_llm_api or args.openai_api_key is not None
    )  # If use_llm_api is True, openai_api_key must be provided
    if args.debug:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG")
        logger.debug("Debug mode is on.")
    else:
        logger.remove()
        logger.add(sys.stdout, level="INFO")

    logger.info("Retrieving Zotero corpus...")
    corpus, all_collection_paths = get_zotero_corpus(args.zotero_id, args.zotero_key)
    logger.info(f"Retrieved {len(corpus)} papers from Zotero.")
    if args.zotero_ignore:
        logger.info(f"Ignoring papers in:\n {args.zotero_ignore}...")
        corpus = filter_corpus(corpus, args.zotero_ignore)
        logger.info(f"Remaining {len(corpus)} papers after filtering.")
    logger.info("Retrieving Arxiv papers...")
    papers = get_arxiv_paper(args.arxiv_query, args.debug)
    if len(papers) == 0:
        logger.info("No new papers found. Yesterday maybe a holiday and no one submit their work :). If this is not the case, please check the ARXIV_QUERY.")
        if not args.send_empty:
          exit(0)
    else:
        if args.use_llm_api:
            logger.info("Using OpenAI API as global LLM.")
            set_global_llm(api_key=args.openai_api_key, base_url=args.openai_api_base, model=args.model_name, lang=args.language)
        else:
            logger.info("Using Local LLM as global LLM.")
            set_global_llm(lang=args.language)

    if os.path.exists('.categories'):
        logger.info("Found .categories file, using grouped recommendations.")
        category_groups = parse_categories('.categories', all_collection_paths)
        seen_ids = set()
        grouped_results = []  # list of (label, [(paper, score)])
        for label, paths, max_num in category_groups:
            subset_corpus = filter_corpus_by_paths(corpus, paths)
            logger.info(f"Category '{label}': {len(subset_corpus)} corpus papers matched.")
            ranked = rerank_paper(papers, subset_corpus)
            group_papers = []
            for p in ranked:
                if p.arxiv_id not in seen_ids:
                    group_papers.append((p, p.score))
                    seen_ids.add(p.arxiv_id)
                if len(group_papers) >= max_num:
                    break
            grouped_results.append((label, group_papers))
        # Restore per-group scores (later rerank_paper calls overwrite .score in-place)
        for label, group in grouped_results:
            for p, score in group:
                p.score = score
        html = render_email_grouped([(label, [p for p, _ in group]) for label, group in grouped_results])
    else:
        logger.info("Reranking papers...")
        papers = rerank_paper(papers, corpus)
        if args.max_paper_num != -1:
            papers = papers[:args.max_paper_num]
        html = render_email(papers)
    logger.info("Sending email...")
    send_email(args.sender, args.receiver, args.sender_password, args.smtp_server, args.smtp_port, html)
    logger.success("Email sent successfully! If you don't receive the email, please check the configuration and the junk box.")

