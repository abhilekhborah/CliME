#!/usr/bin/env python3
import os
import pandas as pd
import hashlib
import requests
from yars import YARS
from utils import display_results, download_image

class RedditImageScraper:
    def __init__(self, output_dir='./reddit_data'):
        """Initialize the Reddit image scraper"""
        self.miner = YARS()
        self.output_dir = output_dir
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        self.images_dir = os.path.join(output_dir, 'images')
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Path for CSV
        self.csv_path = os.path.join(output_dir, 'reddit_posts.csv')
        
        print(f"Reddit Image Scraper initialized. Output directory: {output_dir}")
    
    def _has_image(self, post):
        """Check if a post has an image"""
        # Check various possible image fields in the post data
        if post.get('is_video', False):
            return False
        
        # Check for image URL
        if 'image_url' in post and post['image_url']:
            return True
        
        # Check for URL patterns that suggest images
        if 'url' in post:
            url = post['url'].lower()
            if any(ext in url for ext in ['.jpg', '.jpeg', '.png', '.gif']):
                return True
            if any(domain in url for domain in ['i.redd.it', 'i.imgur.com']):
                return True
        
        # Check for thumbnail
        if 'thumbnail_url' in post and post['thumbnail_url'] and post['thumbnail_url'] != 'self':
            if not post['thumbnail_url'].startswith('https://styles.redditmedia.com'):  # Skip default thumbnails
                return True
        
        return False
    
    def _get_image_url(self, post):
        """Get the best available image URL from a post"""
        if 'image_url' in post and post['image_url']:
            return post['image_url']
        
        if 'url' in post:
            url = post['url']
            if any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif']):
                return url
            if any(domain in url.lower() for domain in ['i.redd.it', 'i.imgur.com']):
                return url
        
        if 'thumbnail_url' in post and post['thumbnail_url'] and post['thumbnail_url'] != 'self':
            if not post['thumbnail_url'].startswith('https://styles.redditmedia.com'):
                return post['thumbnail_url']
        
        return None
    
    def _process_and_save_post(self, post):
        """Process a post and save it if it has an image"""
        if not self._has_image(post):
            return None
        
        # Get the best image URL
        image_url = self._get_image_url(post)
        if not image_url:
            return None
        
        # Generate a unique ID for the post
        post_id = post.get('id', hashlib.md5(post.get('permalink', '').encode()).hexdigest())
        
        # Download image
        image_path = self._download_image(post_id, image_url)
        
        # Prepare data for CSV
        post_data = {
            'text': post.get('title', '') + '\n' + post.get('selftext', ''),
            'username': post.get('author', ''),
            'timestamp': post.get('created_utc', ''),
            'likes': post.get('score', 0),
            'comments': post.get('num_comments', 0),
            'post_url': f"https://www.reddit.com{post.get('permalink', '')}",
            'image_url': image_url,
            'media_paths': image_path if image_path else "",
            'subreddit': post.get('subreddit', ''),
            'post_id': post_id
        }
        
        # Save to CSV
        self._save_to_csv(post_data)
        
        return post_data
    
    def _download_image(self, post_id, img_url):
        """Download an image and return the path if successful"""
        if not img_url:
            return None
        
        try:
            # Generate filename
            filename = f"{post_id}.jpg"
            img_path = os.path.join(self.images_dir, filename)
            
            # Skip if file already exists
            if os.path.exists(img_path):
                return img_path
            
            # Download image
            response = requests.get(img_url, stream=True, timeout=10)
            if response.status_code == 200:
                with open(img_path, 'wb') as f:
                    f.write(response.content)
                return img_path
            else:
                print(f"Failed to download image: HTTP {response.status_code}")
                return None
        except Exception as e:
            print(f"Error downloading image: {str(e)}")
            return None
    
    def _save_to_csv(self, post_data):
        """Save post data to CSV"""
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
    
    def search_reddit_images(self, query, limit=1000, subreddit=None):
        """
        Search Reddit for posts with images matching a query
        
        Parameters:
        - query: Search term
        - limit: Maximum number of posts to collect
        - subreddit: Optional specific subreddit to search in
        
        Returns: List of posts with images
        """
        print(f"Searching Reddit for '{query}' with images (limit: {limit})...")
        
        # Add subreddit prefix if specified
        full_query = f"subreddit:{subreddit} {query}" if subreddit else query
        
        processed_posts = []
        processed_count = 0
        search_batch_size = min(250, limit)  # Reddit API has limitations
        total_searched = 0
        
        # Continue searching in batches until we reach our limit
        while processed_count < limit and total_searched < limit * 5:  # Set a reasonable upper bound
            remaining = limit - processed_count
            batch_size = min(search_batch_size, remaining * 3)  # Get more to filter
            
            print(f"Searching batch of {batch_size} posts (found {processed_count} so far)...")
            
            # Search Reddit
            search_results = self.miner.search_reddit(full_query, limit=batch_size)
            if not search_results:
                print("No more results found.")
                break
                
            total_searched += len(search_results)
            
            # Filter for posts with images and process them
            for post in search_results:
                if processed_count >= limit:
                    break
                    
                processed_post = self._process_and_save_post(post)
                if processed_post:
                    processed_posts.append(processed_post)
                    processed_count += 1
                    print(f"Post {processed_count}/{limit} with image processed")
            
            # If we didn't get any images in this batch, we might need to stop
            if len(processed_posts) == 0 or len(search_results) < batch_size:
                print("No more results with images found.")
                break
        
        print(f"Found {len(processed_posts)} posts with images out of {total_searched} search results")
        return processed_posts
    
    def fetch_subreddit_image_posts(self, subreddit, limit=1000, category="top", time_filter="all"):
        """
        Fetch posts with images from a specific subreddit
        
        Parameters:
        - subreddit: Subreddit name
        - limit: Maximum number of posts to collect
        - category: hot, new, top, rising, controversial
        - time_filter: hour, day, week, month, year, all
        
        Returns: List of posts with images
        """
        print(f"Fetching {category} posts with images from r/{subreddit} (limit: {limit})...")
        
        processed_posts = []
        processed_count = 0
        fetch_batch_size = min(250, limit)  # Reddit API has limitations
        total_fetched = 0
        
        # Continue fetching in batches until we reach our limit
        while processed_count < limit and total_fetched < limit * 5:  # Set a reasonable upper bound
            remaining = limit - processed_count
            batch_size = min(fetch_batch_size, remaining * 3)  # Get more to filter
            
            print(f"Fetching batch of {batch_size} posts (found {processed_count} so far)...")
            
            # Fetch posts from subreddit
            subreddit_posts = self.miner.fetch_subreddit_posts(
                subreddit, 
                limit=batch_size,
                category=category,
                time_filter=time_filter
            )
            
            if not subreddit_posts:
                print("No more posts found.")
                break
                
            total_fetched += len(subreddit_posts)
            
            # Filter for posts with images and process them
            new_processed = 0
            for post in subreddit_posts:
                if processed_count >= limit:
                    break
                    
                processed_post = self._process_and_save_post(post)
                if processed_post:
                    processed_posts.append(processed_post)
                    processed_count += 1
                    new_processed += 1
                    print(f"Post {processed_count}/{limit} with image processed")
            
            # If we didn't get any images in this batch or reached the end, we might need to stop
            if new_processed == 0 or len(subreddit_posts) < batch_size:
                print("No more posts with images found or reached end of available posts.")
                break
        
        print(f"Found {len(processed_posts)} posts with images out of {total_fetched} subreddit posts")
        return processed_posts
    
    def fetch_user_image_posts(self, username, limit=1000):
        """
        Fetch posts with images from a specific user
        
        Parameters:
        - username: Reddit username
        - limit: Maximum number of posts to collect
        
        Returns: List of posts with images
        """
        print(f"Fetching posts with images from u/{username} (limit: {limit})...")
        
        processed_posts = []
        processed_count = 0
        fetch_batch_size = min(250, limit)  # Reddit API has limitations
        total_fetched = 0
        
        # Continue fetching in batches until we reach our limit
        while processed_count < limit and total_fetched < limit * 5:  # Set a reasonable upper bound
            remaining = limit - processed_count
            batch_size = min(fetch_batch_size, remaining * 3)  # Get more to filter
            
            print(f"Fetching batch of {batch_size} user items (found {processed_count} image posts so far)...")
            
            # Fetch user data
            user_data = self.miner.scrape_user_data(username, limit=batch_size)
            
            if not user_data:
                print("No more user data found.")
                break
                
            total_fetched += len(user_data)
            
            # Filter for submissions (posts) with images and process them
            new_processed = 0
            for item in user_data:
                if processed_count >= limit:
                    break
                    
                # Skip comments, only process submissions
                if item.get('type') != 'submission':
                    continue
                    
                processed_post = self._process_and_save_post(item)
                if processed_post:
                    processed_posts.append(processed_post)
                    processed_count += 1
                    new_processed += 1
                    print(f"Post {processed_count}/{limit} with image processed")
            
            # If we didn't get any images in this batch or reached the end, we might need to stop
            if new_processed == 0 or len(user_data) < batch_size:
                print("No more posts with images found or reached end of available posts.")
                break
        
        print(f"Found {len(processed_posts)} posts with images out of {total_fetched} user items")
        return processed_posts

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Scrape Reddit for posts with images')
    parser.add_argument('--query', '-q', type=str, help='Search term(s)')
    parser.add_argument('--subreddit', '-s', type=str, help='Specific subreddit to search (optional)')
    parser.add_argument('--user', '-u', type=str, help='Username to get posts from')
    parser.add_argument('--limit', '-l', type=int, default=1000, help='Maximum number of posts to collect (default: 1000)')
    parser.add_argument('--category', '-c', choices=['hot', 'new', 'top', 'rising', 'controversial'], default='top', 
                        help='Post category - hot, new, top, rising, controversial (default: top)')
    parser.add_argument('--time', '-t', choices=['hour', 'day', 'week', 'month', 'year', 'all'], default='all',
                        help='Time filter - hour, day, week, month, year, all (default: all)')
    parser.add_argument('--output', '-o', default='./reddit_images', help='Output directory path')
    args = parser.parse_args()
    
    # Initialize the scraper
    reddit_scraper = RedditImageScraper(output_dir=args.output)
    collected_posts = []
    
    if args.user:
        # Get image posts from a specific user
        print(f"Fetching posts with images from user u/{args.user}...")
        collected_posts = reddit_scraper.fetch_user_image_posts(args.user, limit=args.limit)
        display_results(collected_posts, f"USER {args.user.upper()} IMAGE POSTS")
    
    elif args.subreddit:
        # Get image posts from a specific subreddit
        print(f"Fetching {args.category} posts with images from r/{args.subreddit}...")
        collected_posts = reddit_scraper.fetch_subreddit_image_posts(
            args.subreddit, 
            limit=args.limit, 
            category=args.category, 
            time_filter=args.time
        )
        display_results(collected_posts, f"r/{args.subreddit.upper()} {args.category.upper()} IMAGES")
    
    elif args.query:
        # Search for image posts by query
        print(f"Searching Reddit for '{args.query}' with images...")
        collected_posts = reddit_scraper.search_reddit_images(args.query, limit=args.limit)
        display_results(collected_posts, f"'{args.query.upper()}' IMAGE SEARCH")
    
    else:
        # If no specific filter, get from r/all to get a broad range of posts
        print(f"Fetching {args.category} posts with images from all of Reddit...")
        collected_posts = reddit_scraper.fetch_subreddit_image_posts(
            "all", 
            limit=args.limit, 
            category=args.category, 
            time_filter=args.time
        )
        display_results(collected_posts, f"ALL REDDIT {args.category.upper()} IMAGES")
    
    print(f"\nTotal posts collected: {len(collected_posts)}")
    print(f"All image posts have been saved to:")
    print(f"- CSV file: {reddit_scraper.csv_path}")
    print(f"- Images folder: {reddit_scraper.images_dir}")
