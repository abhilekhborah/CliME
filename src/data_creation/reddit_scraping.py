import os
import time
import pandas as pd
import argparse
import json
import hashlib
import re
import signal
import sys
import requests
import praw
from datetime import datetime
from urllib.parse import urlparse
from PIL import Image
from io import BytesIO

class RedditClimateScraper:
    def __init__(self, client_id, client_secret, username, password, 
                 output_dir='reddit_climate_data', user_agent='Climate Content Research Bot v1.0'):
        """Initialize the Reddit scraper with login credentials"""
        self.client_id = client_id
        self.client_secret = client_secret
        self.username = username
        self.password = password
        self.user_agent = user_agent
        self.output_dir = output_dir
        self.current_posts_data = []  # Store posts as we collect them
        self.interrupted = False      # Flag for graceful shutdown
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create images directory upfront
        self.images_dir = os.path.join(output_dir, 'images')
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Path for tracking collected posts
        self.tracking_file = os.path.join(output_dir, 'collected_climate_posts.json')
        self.csv_path = os.path.join(output_dir, 'climate_posts.csv')
        
        # Load previously collected post information
        self.collected_posts = self._load_tracking_data()
        
        # Set up signal handler for Ctrl+C
        signal.signal(signal.SIGINT, self._handle_interrupt)
        
        print(f"Reddit scraper initialized. Found {len(self.collected_posts)} previously collected posts.")
    
    def _handle_interrupt(self, sig, frame):
        """Handle keyboard interrupt (Ctrl+C) gracefully"""
        print("\n\nInterrupt received! Saving collected data before exiting...")
        self.interrupted = True
        
        # Save current data
        if self.current_posts_data:
            df = pd.DataFrame(self.current_posts_data)
            
            # Save to CSV
            if not df.empty:
                df.to_csv(self.csv_path, index=False)
                print(f"Saved {len(df)} posts to {self.csv_path}")
            
            # Save tracking data
            self._save_tracking_data()
            print("Tracking data saved")
        
        print("Gracefully shutting down. All data collected so far has been saved.")
        sys.exit(0)
    
    def _load_tracking_data(self):
        """Load tracking data of previously collected posts"""
        if os.path.exists(self.tracking_file):
            try:
                with open(self.tracking_file, 'r') as f:
                    return json.load(f)
            except:
                print("Could not load previous tracking data. Starting fresh.")
                return {}
        return {}
    
    def _save_tracking_data(self):
        """Save tracking data of collected posts"""
        with open(self.tracking_file, 'w') as f:
            json.dump(self.collected_posts, f)
    
    def _get_post_hash(self, post_id, post_title):
        """Generate a unique hash for a post based on content and id"""
        combined = f"{post_id}|{post_title}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def login(self):
        """Log in to Reddit using the provided credentials"""
        try:
            print("Logging in to Reddit...")
            self.reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                username=self.username,
                password=self.password,
                user_agent=self.user_agent
            )
            
            # Verify that we're logged in
            username = self.reddit.user.me().name
            print(f"Successfully logged in as {username}.")
            return True
            
        except Exception as e:
            print(f"Login failed: {str(e)}")
            return False
    
    def _save_post_to_csv(self, post_data):
        """Save a single post to the CSV file"""
        try:
            # If this is the first post, write with header
            if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
                df = pd.DataFrame([post_data])
                df.to_csv(self.csv_path, index=False)
            else:
                # Append without header
                df = pd.DataFrame([post_data])
                df.to_csv(self.csv_path, mode='a', header=False, index=False)
            return True
        except Exception as e:
            print(f"Error saving post to CSV: {str(e)}")
            return False
    
    def _download_image(self, post_hash, img_url):
        """Download a single image and return the path if successful"""
        if not img_url:
            return None
        
        try:
            # Determine file extension from URL or default to jpg
            parsed_url = urlparse(img_url)
            path = parsed_url.path.lower()
            
            if path.endswith('.jpg') or path.endswith('.jpeg'):
                ext = '.jpg'
            elif path.endswith('.png'):
                ext = '.png'
            elif path.endswith('.gif') and not self._is_animated_gif(img_url):
                ext = '.gif'
            else:
                ext = '.jpg'  # Default
            
            # Generate filename from hash
            filename = f"{post_hash}{ext}"
            img_path = os.path.join(self.images_dir, filename)
            
            # Skip if file already exists
            if os.path.exists(img_path):
                return img_path
            
            # Download image
            response = requests.get(img_url, stream=True, timeout=10, headers={'User-Agent': self.user_agent})
            if response.status_code == 200:
                with open(img_path, 'wb') as f:
                    f.write(response.content)
                return img_path
            else:
                print(f"Failed to download image for {post_hash}: HTTP {response.status_code}")
                return None
        except Exception as e:
            print(f"Error downloading image for {post_hash}: {str(e)}")
            return None
    
    def _is_animated_gif(self, url):
        """Check if a GIF is animated (we want to skip these)"""
        try:
            response = requests.get(url, timeout=5, headers={'User-Agent': self.user_agent})
            img = Image.open(BytesIO(response.content))
            
            try:
                img.seek(1)  # Try to move to the second frame
                return True   # If we can, it's animated
            except EOFError:
                return False  # If we can't, it's not animated
        except:
            return True  # In case of error, assume it's animated to be safe
    
    def is_climate_related(self, title, selftext='', less_strict=True):
        """Check if content contains actual climate-related content using multi-layered verification"""
        # Combine title and selftext for checking
        text = (title + ' ' + selftext).lower()
        
        # 1. Core climate science/policy keywords
        science_keywords = {
            'climate change', 'global warming', 'carbon', 'emission', 'greenhouse gas', 
            'ipcc', 'sea level', 'renewable energy', 'fossil fuel', 'paris agreement', 
            'extreme weather', 'drought', 'flood', 'methane', 'net zero', 'decarbon', 
            'climate crisis', 'anthropocene', 'tipping point', 'climate model', 
            'climate science', 'climate data', 'climate impact', 'climate disaster',
            'climate risk', 'climate policy', 'climate legislation', 'climate denial',
            'heat wave', 'green energy', 'clean energy', 'co2', 'esg', 'sustainability'
        }
        
        # 2. Environmental impact keywords
        impact_keywords = {
            'biodiversity loss', 'ecosystem collapse', 'arctic melt', 'glacial retreat',
            'coral bleaching', 'wildfire', 'heatwave', 'hurricane', 'typhoon', 'cyclone',
            'air quality', 'ocean acidification', 'permafrost', 'species extinction',
            'food security', 'climate refugee', 'climate migration', 'pollution',
            'environment', 'recycling', 'clean air', 'clean water', 'solar', 'wind power'
        }
        
        # 3. Climate solutions/technology
        solution_keywords = {
            'solar power', 'wind energy', 'ev transition', 'carbon capture',
            'ccs', 'geoengineering', 'climate tech', 'sustainable', 'decarbonization',
            'carbon tax', 'carbon price', 'climate finance', 'green investment',
            'climate adaptation', 'mitigation', 'resilience', 'electric vehicle',
            'green tech', 'clean tech', 'esg', 'tesla', 'solar panel'
        }
        
        # Negative keywords to exclude non-climate posts
        exclusion_patterns = {
            r'\b(stocks|crypto|nft|ethereum|bitcoin|dividend|airdrop)\b',
            r'\b(price prediction|forecast|buy now|trending|investment advice)\b'
        }
        
        # Check for exclusion patterns
        for pattern in exclusion_patterns:
            if re.search(pattern, text):
                return False
        
        # Combine all climate-related patterns
        climate_patterns = (
            science_keywords | 
            impact_keywords | 
            solution_keywords
        )
        
        # Check for climate references
        matches = set()
        for pattern in climate_patterns:
            if re.search(r'\b' + re.escape(pattern) + r'\b', text):
                matches.add(pattern)
                # Only require 1 match if less_strict is True
                if less_strict or len(matches) >= 2:
                    return True
        
        return False
        
    def _extract_image_url(self, post):
        """Extract image URL from a Reddit post"""
        # Case 1: Direct image post
        if hasattr(post, 'url'):
            url = post.url.lower()
            if any(url.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif']):
                return post.url
        
        # Case 2: Reddit gallery
        if hasattr(post, 'is_gallery') and post.is_gallery:
            try:
                # Get the first image from the gallery
                media_metadata = post.media_metadata
                for media_id in media_metadata:
                    item = media_metadata[media_id]
                    if item['e'] == 'Image':
                        largest_src = item['s']
                        return largest_src['u']
            except:
                pass
                
        # Case 3: Image in post.preview
        if hasattr(post, 'preview'):
            try:
                return post.preview['images'][0]['source']['url']
            except:
                pass
                
        # Case 4: Embedded media
        if hasattr(post, 'media') and post.media:
            try:
                if 'oembed' in post.media and 'thumbnail_url' in post.media['oembed']:
                    return post.media['oembed']['thumbnail_url']
            except:
                pass
                
        # No image found
        return None
        
    def scrape_subreddits(self, subreddits, search_terms=None, max_posts=100, time_filter='all'):
        """
        Scrape climate-related posts with images from specified subreddits
        
        Parameters:
        - subreddits: List of subreddit names to search (without 'r/')
        - search_terms: Optional list of terms to search for within the subreddits
        - max_posts: Maximum number of posts to collect
        - time_filter: 'all', 'day', 'month', 'week', 'year'
        
        Returns: DataFrame of scraped climate posts with images
        """
        try:
            # Calculate how many more posts we need to collect
            total_collected = len(self.collected_posts)
            remaining_to_collect = max(0, max_posts - total_collected)
            
            if remaining_to_collect <= 0:
                print(f"Already collected {total_collected} posts, which meets or exceeds the target of {max_posts}.")
                # Load existing data and return
                return self._load_existing_data()
            
            print(f"Already collected {total_collected} posts. Need {remaining_to_collect} more to reach target of {max_posts}.")
            
            # Combine search terms into a query string
            search_query = ' OR '.join(search_terms) if search_terms else None
            
            # Load any existing data
            posts_data = []
            existing_df = self._load_existing_data()
            if not existing_df.empty:
                posts_data.extend(existing_df.to_dict('records'))
            
            new_posts_count = 0
            for subreddit_name in subreddits:
                if self.interrupted or new_posts_count >= remaining_to_collect:
                    break
                    
                try:
                    print(f"Searching subreddit: r/{subreddit_name}...")
                    subreddit = self.reddit.subreddit(subreddit_name)
                    
                    # Get submissions - either search or get submissions
                    if search_query:
                        submissions = subreddit.search(search_query, time_filter=time_filter, limit=None)
                    else:
                        # If no search terms, get top posts
                        submissions = subreddit.top(time_filter=time_filter, limit=None)
                    
                    # Process submissions
                    for post in submissions:
                        if self.interrupted or new_posts_count >= remaining_to_collect:
                            break
                            
                        try:
                            # Skip if already collected
                            post_hash = self._get_post_hash(post.id, post.title)
                            if post_hash in self.collected_posts:
                                continue
                                
                            # Check if climate-related
                            if not self.is_climate_related(post.title, getattr(post, 'selftext', ''), less_strict=True):
                                continue
                                
                            # Extract image URL
                            img_url = self._extract_image_url(post)
                            if not img_url:
                                continue  # Skip posts without images
                            
                            # Get post details
                            post_data = {
                                'post_id': post.id,
                                'post_title': post.title,
                                'subreddit': post.subreddit.display_name,
                                'author': str(post.author),
                                'created_utc': datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                                'score': post.score,
                                'upvote_ratio': getattr(post, 'upvote_ratio', 0),
                                'num_comments': post.num_comments,
                                'permalink': f"https://www.reddit.com{post.permalink}",
                                'image_url': img_url,
                                'post_hash': post_hash,
                                'is_self': post.is_self,
                                'selftext': getattr(post, 'selftext', '')[:500]  # Truncate long text
                            }
                            
                            # Save post to CSV
                            self._save_post_to_csv(post_data)
                            
                            # Download image
                            image_path = self._download_image(post_hash, img_url)
                            
                            # Add to tracking
                            self.collected_posts[post_hash] = {
                                'collected_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                                'post_id': post.id,
                                'image_downloaded': image_path is not None,
                                'image_path': image_path
                            }
                            
                            # Add to current data list
                            posts_data.append(post_data)
                            self.current_posts_data.append(post_data)
                            
                            # Update counter
                            new_posts_count += 1
                            
                            # Save tracking data after each post
                            self._save_tracking_data()
                            
                            # Print progress
                            print(f"Post {new_posts_count}/{remaining_to_collect} collected from r/{subreddit_name} - Image: {'✓' if image_path else '✗'}")
                            
                            # Sleep briefly to avoid hitting rate limits
                            time.sleep(0.5)
                            
                        except Exception as e:
                            print(f"Error processing post: {str(e)}")
                            continue
                            
                    print(f"Finished searching r/{subreddit_name}. Found {new_posts_count} climate posts with images so far.")
                    
                except Exception as e:
                    print(f"Error accessing subreddit r/{subreddit_name}: {str(e)}")
                    continue
            
            print(f"Scraping completed. Collected {new_posts_count} new posts ({total_collected + new_posts_count} total).")
            
            # Convert to DataFrame
            df = pd.DataFrame(posts_data)
            return df
                
        except Exception as e:
            print(f"Error during scraping: {str(e)}")
            return self._load_existing_data()  # Return existing data in case of error
    
    def _load_existing_data(self):
        """Load existing post data from CSV if available"""
        if os.path.exists(self.csv_path):
            try:
                return pd.read_csv(self.csv_path)
            except:
                print("Could not load existing CSV data.")
        return pd.DataFrame()
    
    def save_data(self, df):
        """Save the scraped data to CSV"""
        if df.empty:
            print("No data to save.")
            return
        
        # Save DataFrame to CSV (this should already be done incrementally,
        # but we do it again as a backup)
        df.to_csv(self.csv_path, index=False)
        print(f"Data saved to {self.csv_path}")
        
        # Count images already downloaded
        downloaded_count = sum(1 for post_hash in self.collected_posts 
                              if self.collected_posts[post_hash].get('image_downloaded', False))
        
        print(f"Total of {downloaded_count} images saved to {self.images_dir}")
        print(f"All data and images saved to {self.output_dir}")

def main():
    """Main function to run the Reddit climate scraper"""
    parser = argparse.ArgumentParser(description='Scrape climate-related posts with images from Reddit')
    
    parser.add_argument('--client_id', '-c', required=True, help='Reddit API client ID')
    parser.add_argument('--client_secret', '-s', required=True, help='Reddit API client secret')
    parser.add_argument('--username', '-u', required=True, help='Reddit username')
    parser.add_argument('--password', '-p', required=True, help='Reddit password')
    parser.add_argument('--max_posts', '-m', type=int, default=100, help='Maximum number of posts to collect')
    parser.add_argument('--subreddits', '-r', nargs='+', default=['environment', 'climate', 'ClimateChange', 'GlobalWarming', 'sustainability'], 
                        help='List of subreddits to search (without r/)')
    parser.add_argument('--search_terms', '-t', nargs='+', 
                        default=['climate change', 'global warming', 'renewable energy', 'carbon emissions'],
                        help='Terms to search for within the subreddits')
    parser.add_argument('--time_filter', '-f', choices=['all', 'day', 'week', 'month', 'year'], default='all',
                        help='Time filter for posts')
    parser.add_argument('--output', '-o', default='reddit_climate_data', help='Output directory path')
    
    args = parser.parse_args()
    
    print("\n--- Real-time Reddit Climate Image Scraper ---")
    print("Press Ctrl+C at any time to save & exit gracefully")
    print("All posts and images will be saved in real-time\n")
    
    # Initialize scraper
    scraper = RedditClimateScraper(
        client_id=args.client_id,
        client_secret=args.client_secret,
        username=args.username,
        password=args.password,
        output_dir=args.output
    )
    
    try:
        # Login to Reddit
        if not scraper.login():
            print("Login failed. Exiting.")
            return
        
        # Scrape climate posts
        df = scraper.scrape_subreddits(
            subreddits=args.subreddits,
            search_terms=args.search_terms,
            max_posts=args.max_posts,
            time_filter=args.time_filter
        )
        
        # Save any final data
        scraper.save_data(df)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        # Even on error, try to save what we have
        try:
            if hasattr(scraper, 'current_posts_data') and scraper.current_posts_data:
                df = pd.DataFrame(scraper.current_posts_data)
                scraper.save_data(df)
                print("Saved partial data despite error.")
        except:
            pass

if __name__ == "__main__":
    main()