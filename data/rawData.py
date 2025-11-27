'''
This file contains code to construct a raw data set emassing data over an extended period of time.
It tries to preselect only where its necessary due to data or api concerns.
All subsets of Data should be constructed from this master data set.
Data Source is the tmdb api. It has a rate limit of 40 requests per second
"""

'''

import requests
import time
import json
import os

TMDB_API_KEY = "50fabac2bdba5ae592a69cf47567f3f4"
BASE_URL = "https://api.themoviedb.org/3"


def fetch_tmdb_movies(start_date, end_date):
    """
    Fetches a list of movies from TMDB within a date range.
    Handles pagination to get as many results as the API allows (up to 500 pages).
    """
    movies = []
    page = 1
    total_pages = 1  # Initialize to 1 to enter the loop
    total_pages = 1  # Initialize to 1 to enter the loop

    while page <= total_pages:
        url = (
            f"{BASE_URL}/discover/movie"
            f"?api_key={TMDB_API_KEY}"
            f"&primary_release_date.gte={start_date}"
            f"&primary_release_date.lte={end_date}"
            f"&page={page}"
            f"&sort_by=revenue.desc"
        )
        url = (
            f"{BASE_URL}/discover/movie"
            f"?api_key={TMDB_API_KEY}"
            f"&primary_release_date.gte={start_date}"
            f"&primary_release_date.lte={end_date}"
            f"&page={page}"
            f"&sort_by=revenue.desc"
        )
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                page_movies = data.get("results", [])
                if not page_movies:  # Stop if a page is empty
                    print(
                        f"Page {page} was empty. Assuming end of results for this query."
                    )
                if not page_movies:  # Stop if a page is empty
                    print(
                        f"Page {page} was empty. Assuming end of results for this query."
                    )
                    break
                movies.extend(data.get("results", []))
                # The API limits pages to 500.
                total_pages = min(data.get("total_pages", 0), 500)
                print(f"Fetched page {page}/{total_pages}...")
                page += 1
            else:
                print(f"Error fetching page {page}: Status code {resp.status_code}")
                break
            # Respect the rate limit
            time.sleep(0.03)
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            break
    return movies


def fetch_tmdb_details(movie_id):
    """Fetches detailed info for a specific movie ID."""
    url = f"{BASE_URL}/movie/{movie_id}?api_key={TMDB_API_KEY}&append_to_response=credits,keywords"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            print(
                f"Error fetching details for movie ID {movie_id}: Status code {resp.status_code}"
            )
            print(
                f"Error fetching details for movie ID {movie_id}: Status code {resp.status_code}"
            )
            return None
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"Request for movie ID {movie_id} failed: {e}")
        return None


def save_to_json(data, filename):
    """Saves data to a JSON file."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    # This print can be too verbose when saving after every movie, so it's moved to the main loop
    print(f"Data successfully saved to {filename}. Total movies: {len(data)}.")


def load_existing_data(filename):
    """Loads existing movie data from a JSON file if it exists."""
    movies = []
    try:
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    movies.append(json.loads(line))
            return movies
    except FileNotFoundError:
        # If file doesn't exist, start fresh
        print("No existing data file found or file is invalid. Starting fresh.")
        return []


def main():
    """
    Main function to fetch movie data and save it.
    Define your desired date range here.
    """
    # For a multi-year analysis around COVID-19, you might use a range like this.
    start_year = 2015
    end_year = 2024

    # Save files in the same directory as the script.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    discovery_cache_filename = os.path.join(script_dir, "tmdb_movies_master.cache.json")
    output_filename = os.path.join(script_dir, "tmdb_movies_master.jsonl")

    all_movies_discovered = {}
    resume_year = start_year

    # --- Step 1: Resiliently discover all movie IDs year by year ---
    try:
        with open(discovery_cache_filename, "r", encoding="utf-8") as f:
            cache = json.load(f)
            resume_year = cache.get("last_completed_year", start_year - 1) + 1
            resume_year = cache.get("last_completed_year", start_year - 1) + 1
            # Load all previously discovered movies to avoid data loss on resume
            all_movies_discovered = cache.get(
                "discovered_movies", all_movies_discovered
            )
            all_movies_discovered = cache.get(
                "discovered_movies", all_movies_discovered
            )
            # JSON stores keys as strings, so convert them back to integers
            all_movies_discovered = {
                int(k): v for k, v in all_movies_discovered.items()
            }
            all_movies_discovered = {
                int(k): v for k, v in all_movies_discovered.items()
            }
            print(f"Loaded discovery cache. Resuming from year {resume_year}.")
    except (FileNotFoundError, json.JSONDecodeError):
        print("No discovery cache found. Starting from the beginning.")

    for year in range(resume_year, end_year + 1):
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        print(f"\n--- Fetching movie list for {year} ---")
        movies_for_year = fetch_tmdb_movies(start_date, end_date)

        for movie in movies_for_year:
            movie_id = movie.get("id")
            if movie_id and movie_id not in all_movies_discovered:
                all_movies_discovered[movie_id] = movie

        print(
            f"Found {len(movies_for_year)} movies for {year}. Total unique movies so far: {len(all_movies_discovered)}"
        )

        print(
            f"Found {len(movies_for_year)} movies for {year}. Total unique movies so far: {len(all_movies_discovered)}"
        )

        # Cache the progress after each year is done
        with open(discovery_cache_filename, "w", encoding="utf-8") as f:
            cache_data = {
                "last_completed_year": year,
                "discovered_movies": all_movies_discovered,
            }
        with open(discovery_cache_filename, "w", encoding="utf-8") as f:
            cache_data = {
                "last_completed_year": year,
                "discovered_movies": all_movies_discovered,
            }
            json.dump(cache_data, f, indent=4)
        print(f"Saved discovery progress for year {year}.")

    unique_movies_list = list(all_movies_discovered.values())
    # Sort the movies by ID to ensure a consistent processing order every time.
    # This is critical for the resume logic to appear sequential.
    unique_movies_list = sorted(
        list(all_movies_discovered.values()), key=lambda m: m["id"]
    )
    unique_movies_list = sorted(
        list(all_movies_discovered.values()), key=lambda m: m["id"]
    )

    print(
        f"\n--- Finished discovery. Found a total of {len(unique_movies_list)} unique movies between {start_year} and {end_year}. ---"
    )
    print(
        f"\n--- Finished discovery. Found a total of {len(unique_movies_list)} unique movies between {start_year} and {end_year}. ---"
    )

    # --- Step 2: Fetch details for all unique movies ---
    # Load already fetched data to avoid re-fetching
    all_movie_details = load_existing_data(output_filename)
    existing_ids = {movie["id"] for movie in all_movie_details}
    existing_ids = {movie["id"] for movie in all_movie_details}
    print(f"Found {len(existing_ids)} movies already in the local dataset.")

    # --- Resume logic for detail fetching ---
    movies_to_process = unique_movies_list
    if all_movie_details:
        # Sort existing details by ID to find the last one reliably
        all_movie_details_sorted = sorted(all_movie_details, key=lambda m: m["id"])
        last_fetched_id = all_movie_details_sorted[-1]["id"]

        # Find the index of the last fetched movie in the full list
        try:
            # The list is already sorted by ID
            last_index = [m["id"] for m in unique_movies_list].index(last_fetched_id)
            # Slice the list to only process movies that haven't been fetched yet
            movies_to_process = unique_movies_list[last_index + 1 :]
            print(
                f"Resuming detail fetch. Last fetched ID was {last_fetched_id}. {len(movies_to_process)} movies remaining."
            )
        except ValueError:
            # This case is unlikely if the datasets are consistent but is good for robustness
            print(
                "Warning: Last fetched movie ID not found in the discovered list. Processing all non-existing movies."
            )
            movies_to_process = [
                m for m in unique_movies_list if m["id"] not in existing_ids
            ]

    total_movies_in_master_list = len(unique_movies_list)
    # Iterate only over the movies that need to be processed.
    # The start index for enumerate is adjusted to keep the progress counter accurate.
    for i, movie in enumerate(movies_to_process, start=len(all_movie_details)):
        movie_id = movie.get("id")
        if not movie_id:
            continue

        # The progress counter (i+1) now reflects the position in the overall master list.
        print(
            f"Fetching details for movie {i+1}/{total_movies_in_master_list} (ID: {movie_id})..."
        )
        details = fetch_tmdb_details(movie_id)
        if details:
            # --- Step 3: Append the new movie to the file ---
            # This is much faster than rewriting the whole file every time.
            with open(output_filename, "a", encoding="utf-8") as f:
                f.write(json.dumps(details, ensure_ascii=False) + "\n")
            print(
                f"Saved details for movie ID {movie_id}. Total movies now in file: {i + 1}."
            )
        # Respect the rate limit
        time.sleep(0.025)


if __name__ == "__main__":
    main()
