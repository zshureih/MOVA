import requests
from bs4 import BeautifulSoup
import base64
import os
import re

def get_track_preview_url(track_name, artist):
    url = "https://api.deezer.com/search"

    # Query parameters
    query = f"{track_name} {artist}"
    params = {"q": query, "limit": 1}  # Get only the top result

    # Make the API request
    response = requests.get(url, params=params)

    # Handle the response
    preview_url = None
    if response.status_code == 200:
        data = response.json()
        if data["data"]:
            track = data["data"][0]
            track_name = track["title"]
            artist_name = track["artist"]["name"]
            preview_url = track["preview"]

            print(f"Track: {track_name} by {artist_name}")
            print(f"Preview URL: {preview_url}")
        else:
            print("No tracks found for the query.")
    else:
        print(f"Error: {response.status_code} - {response.text}")

    return preview_url

def download_track_preview(preview_url):
    """Download the track preview from the given URL.
    
    Args:
        preview_url (str): The URL of the track preview.
    
    Returns:
        str: The path to the downloaded track preview.
    """

    # Download the track preview
    response = requests.get(preview_url)

    download_path = "./assets/track_preview.mp3"

    if response.status_code == 200:
        with open(download_path, "wb") as f:
            f.write(response.content)
        print("Track preview downloaded successfully!")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    
    return download_path

def get_audio(artist, title):
    """Get the audio preview of a song by the given artist and title.
    
    Args:
        artist (str): The name of the artist.
        title (str): The title of the song.
        
    Returns:
        str: The path of the downloaded audio preview.
    """
    # token = get_spotify_access_token()
    preview_url = get_track_preview_url(title, artist)
    download_path = download_track_preview(preview_url)
    return download_path

def search_song_lyrics(song_title, artist_name, access_token):
    """
    Search for the lyrics of a song using the Genius API.
    
    Args:
        song_title (str): The title of the song.
        artist_name (str): The name of the artist.
        access_token (str): The Genius API access token.
        
    Returns:
        str: The URL of the song lyrics.
    """
    base_url = "https://api.genius.com"
    headers = {"Authorization": f"Bearer {access_token}"}
    search_url = f"{base_url}/search"
    params = {"q": f"{song_title} {artist_name}"}
    response = requests.get(search_url, headers=headers, params=params)
    
    if response.status_code == 200:
        hits = response.json().get("response", {}).get("hits", [])
        if hits:
            # Assume the first result is the most relevant
            song_url = hits[0]["result"]["url"]
            return song_url
        else:
            return "No results found."
    else:
        return f"Error: {response.status_code}, {response.text}"

def scrape_lyrics(url):
    """Scrape the lyrics of a song from a Genius URL.
    
    Args:
        url (str): The URL of the song lyrics.
    
    Returns:
        str: The lyrics of the song.
    """
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        lyrics_div = soup.find("div", class_="lyrics") or soup.find("div", {"data-lyrics-container": "true"})
        if lyrics_div:
            return lyrics_div.get_text(strip=True, separator="\n")
        else:
            return "Lyrics not found."
    else:
        return f"Error: {response.status_code}"

def get_lyrics(artist, title):
    """Get the lyrics of a song by the given artist and title.
    
    Args:
        artist (str): The name of the artist.
        title (str): The title of the song.
        
    Returns:
        str: The lyrics of the song.
    """
    access_token = os.environ.get("GENIUS_ACCESS_TOKEN")
    song_url = search_song_lyrics(title, artist, access_token)
    lyrics = scrape_lyrics(song_url)
    pattern = r"(?=\[.*?\])"

    # Splitting the string
    sections = re.split(pattern, lyrics)
    return sections[1:]

if __name__ == "__main__":
    artist = "Adele"
    title = "Rolling in the Deep"
    lyrics = get_lyrics(artist, title)
    audio_path = get_audio(artist, title)