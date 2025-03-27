import os
import time
import pandas as pd
import argparse
import json
import hashlib
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

class TwitterClimateScraper:
    def __init__(self, username, password, output_dir='/Volumes/T7/Climmeme/twitter_data', headless=False):
        """Initialize the Twitter scraper with login credentials"""
        self.username = username
        self.password = password
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Path for tracking collected tweets
        self.tracking_file = os.path.join(output_dir, 'collected_climate_tweets.json')
        
        # Load previously collected tweet information
        self.collected_tweets = self._load_tracking_data()
        
        # Configure Chrome options
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--lang=en")
        chrome_options.add_argument("--window-size=1920,1080")
        
        # Initialize the Chrome driver
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )
        self.wait = WebDriverWait(self.driver, 15)
        
        print(f"Twitter scraper initialized. Found {len(self.collected_tweets)} previously collected tweets.")
    
    def _load_tracking_data(self):
        """Load tracking data of previously collected tweets"""
        if os.path.exists(self.tracking_file):
            try:
                with open(self.tracking_file, 'r') as f:
                    return json.load(f)
            except:
                print("Could not load previous tracking data. Starting fresh.")
                return {}
        return {}
    
    def _save_tracking_data(self):
        """Save tracking data of collected tweets"""
        with open(self.tracking_file, 'w') as f:
            json.dump(self.collected_tweets, f)
    
    def _get_tweet_hash(self, tweet_text, user_info):
        """Generate a unique hash for a tweet based on content and user"""
        # Combine tweet text and user info to create a unique identifier
        combined = f"{tweet_text}|{user_info}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def login(self):
        """Log in to Twitter using the provided credentials"""
        try:
            print("Logging in to Twitter...")
            self.driver.get("https://twitter.com/login")
            time.sleep(3)
            
            # Enter username
            username_field = self.wait.until(
                EC.presence_of_element_located((By.XPATH, "//input[@name='text']"))
            )
            username_field.send_keys(self.username)
            
            # Click Next button
            next_button = self.wait.until(
                EC.element_to_be_clickable((By.XPATH, "//span[contains(text(),'Next')]"))
            )
            next_button.click()
            time.sleep(2)
            
            # Enter password
            password_field = self.wait.until(
                EC.presence_of_element_located((By.XPATH, "//input[@name='password']"))
            )
            password_field.send_keys(self.password)
            
            # Click Login button
            login_button = self.wait.until(
                EC.element_to_be_clickable((By.XPATH, "//span[contains(text(),'Log in')]"))
            )
            login_button.click()
            
            # Wait for login to complete
            time.sleep(5)
            print("Successfully logged in to Twitter.")
            return True
            
        except Exception as e:
            print(f"Login failed: {str(e)}")
            return False
    
    def build_search_query(self, search_terms=None, hashtags=None, 
                           from_accounts=None, start_date=None, end_date=None):
        """
        Build a Twitter search query URL with provided parameters
        
        Parameters:
        - search_terms: List of terms to search for
        - hashtags: List of hashtags to include
        - from_accounts: List of accounts to search from
        - start_date: Search start date in format YYYY-MM-DD
        - end_date: Search end date in format YYYY-MM-DD
        
        Returns: Twitter search URL
        """
        base_url = "https://twitter.com/search?q="
        query_parts = []
        
        # Add search terms if provided
        if search_terms:
            term_query = " OR ".join([f'"{term}"' for term in search_terms])
            query_parts.append(f"({term_query})")
        
        # Add hashtags if provided
        if hashtags:
            hashtag_query = " OR ".join([f'#{tag}' for tag in hashtags])
            query_parts.append(f"({hashtag_query})")
        
        # Add from accounts if provided
        if from_accounts:
            from_query = " OR ".join([f'from:{account}' for account in from_accounts])
            query_parts.append(f"({from_query})")
        
        # Add filter for tweets with images
        query_parts.append("filter:images")
        
        # Combine all query parts with AND
        full_query = " ".join(query_parts)
        
        # Add date range if provided
        if start_date and end_date:
            full_query += f" since:{start_date} until:{end_date}"
        
        # Complete the search URL
        search_url = f"{base_url}{full_query}&src=typed_query&f=live"
        
        print(f"Search URL: {search_url}")
        return search_url
    
    def scrape_tweets(self, search_url, max_tweets=100, scroll_pause=2):
        """
        Scrape tweets based on the search URL with enhanced filtering for:
        1. Image-only content (no videos or GIFs)
        2. Genuine climate-related content using multi-layer verification
        
        Parameters:
        - search_url: Twitter search URL to scrape
        - max_tweets: Maximum number of tweets to collect
        - scroll_pause: Pause duration between scrolls in seconds
        
        Returns: DataFrame of scraped tweets
        """
        try:
            # Calculate how many more tweets we need to collect
            total_collected = len(self.collected_tweets)
            remaining_to_collect = max(0, max_tweets - total_collected)
            
            if remaining_to_collect <= 0:
                print(f"Already collected {total_collected} tweets, which meets or exceeds the target of {max_tweets}.")
                # Load existing data and return
                return self._load_existing_data()
            
            print(f"Already collected {total_collected} tweets. Need {remaining_to_collect} more to reach target of {max_tweets}.")
            print(f"Starting to scrape tweets...")
            self.driver.get(search_url)
            time.sleep(5)  # Wait for initial page load
            
            tweets_data = []
            scroll_count = 0
            max_scrolls = 200  # Safety limit
            new_tweets_count = 0
            no_new_tweets_count = 0
            
            # Load any existing data
            existing_df = self._load_existing_data()
            if not existing_df.empty:
                tweets_data.extend(existing_df.to_dict('records'))
            
            while new_tweets_count < remaining_to_collect and scroll_count < max_scrolls:
                # Find all tweet articles on the page
                articles = self.driver.find_elements(By.XPATH, "//article[@data-testid='tweet']")
                
                new_tweets_in_this_scroll = 0
                
                for article in articles:
                    try:
                        # Check if the tweet has media elements
                        img_elements = article.find_elements(By.CLASS_NAME, 'css-9pa8cd')
                        
                        # Skip if no media or only one element (likely profile picture)
                        if len(img_elements) <= 1:
                            continue
                        
                        # Check if this is a video tweet (which we want to exclude)
                        video_elements = article.find_elements(By.TAG_NAME, 'video')
                        video_containers = article.find_elements(By.CSS_SELECTOR, '[data-testid="videoPlayer"]')
                        gif_containers = article.find_elements(By.CSS_SELECTOR, '[data-testid="gifPlayer"]')
                        
                        # Skip if this tweet contains video content
                        if len(video_elements) > 0 or len(video_containers) > 0 or len(gif_containers) > 0:
                            continue
                        
                        # Check that the second media element is actually an image
                        try:
                            img_url = img_elements[1].get_attribute('src')
                            # Only keep tweets with image URLs that look like images
                            if not img_url or not any(ext in img_url.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                                # If URL doesn't end with image extension, check if it's a Twitter image URL pattern
                                if not ('twimg.com' in img_url and '/media/' in img_url):
                                    continue
                        except:
                            continue
                        
                        # Extract tweet text
                        try:
                            tweet_text = article.find_element(By.XPATH, ".//div[@data-testid='tweetText']").text
                        except NoSuchElementException:
                            # Some tweets might not have text
                            tweet_text = ""
                        
                        # Skip tweets that don't contain actual climate-related content
                        if not self.is_climate_related(tweet_text):
                            continue
                        
                        # Extract user information
                        user_info = article.find_element(By.XPATH, ".//div[@data-testid='User-Name']").text
                        
                        # Generate a unique hash for this tweet
                        tweet_hash = self._get_tweet_hash(tweet_text, user_info)
                        
                        # Skip if we've already collected this tweet
                        if tweet_hash in self.collected_tweets:
                            continue
                        
                        # Extract timestamp
                        timestamp = article.find_element(By.XPATH, ".//time").get_attribute('datetime')
                        date = timestamp.split('T')[0]
                        
                        # Try to get tweet ID from the article
                        tweet_id = None
                        try:
                            article_link = article.find_element(By.XPATH, ".//a[contains(@href, '/status/')]")
                            href = article_link.get_attribute('href')
                            tweet_id = href.split('/status/')[1].split('/')[0]
                        except:
                            pass
                        
                        # Extract engagement metrics if available
                        engagement = {}
                        try:
                            replies = article.find_element(By.XPATH, ".//div[@data-testid='reply']").text
                            engagement['replies'] = replies if replies else "0"
                        except:
                            engagement['replies'] = "0"
                            
                        try:
                            retweets = article.find_element(By.XPATH, ".//div[@data-testid='retweet']").text
                            engagement['retweets'] = retweets if retweets else "0"
                        except:
                            engagement['retweets'] = "0"
                            
                        try:
                            likes = article.find_element(By.XPATH, ".//div[@data-testid='like']").text
                            engagement['likes'] = likes if likes else "0"
                        except:
                            engagement['likes'] = "0"
                        
                        # Add tweet to dataset
                        tweet_data = {
                            'tweet_id': tweet_id,
                            'tweet_text': tweet_text,
                            'image_url': img_url,
                            'user_info': user_info,
                            'date': date,
                            'timestamp': timestamp,
                            'engagement': engagement,
                            'tweet_hash': tweet_hash
                        }
                        
                        tweets_data.append(tweet_data)
                        
                        # Add to collected tweets tracking
                        self.collected_tweets[tweet_hash] = {
                            'collected_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'tweet_id': tweet_id,
                            'image_downloaded': False
                        }
                        
                        new_tweets_count += 1
                        new_tweets_in_this_scroll += 1
                        
                        if new_tweets_count >= remaining_to_collect:
                            break
                    
                    except Exception as e:
                        print(f"Error processing tweet: {str(e)}")
                        continue
                
                # Save progress after each scroll to avoid losing data
                if new_tweets_in_this_scroll > 0:
                    self._save_tracking_data()
                    no_new_tweets_count = 0
                    print(f"Collected {new_tweets_count} new tweets ({total_collected + new_tweets_count} total)")
                else:
                    no_new_tweets_count += 1
                
                if new_tweets_count >= remaining_to_collect:
                    break
                
                # If we've scrolled 5 times without finding new tweets, we might be at the end
                if no_new_tweets_count >= 5:
                    print("No new tweets found in the last 5 scrolls. Stopping.")
                    break
                
                # Scroll down to load more tweets
                self.driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
                time.sleep(scroll_pause)
                scroll_count += 1
                
                # Print progress every 10 scrolls
                if scroll_count % 10 == 0:
                    print(f"Scrolled {scroll_count} times. Collected {new_tweets_count} new tweets so far.")
            
            print(f"Scraping completed. Collected {new_tweets_count} new tweets ({total_collected + new_tweets_count} total).")
            
            # Convert to DataFrame
            df = pd.DataFrame(tweets_data)
            return df
            
        except Exception as e:
            print(f"Error during scraping: {str(e)}")
            return self._load_existing_data()  # Return existing data in case of error
    
    def _load_existing_data(self):
        """Load existing tweet data from CSV if available"""
        csv_path = os.path.join(self.output_dir, 'climate_tweets.csv')
        if os.path.exists(csv_path):
            try:
                return pd.read_csv(csv_path)
            except:
                print("Could not load existing CSV data.")
        return pd.DataFrame()
    
    def is_climate_related(self, text):
        """Check if text contains actual climate-related content using multi-layered verification"""
        text_lower = text.lower()
        
        # 1. Core climate science/policy keywords
        science_keywords = {
            'climate change', 'global warming', 'carbon', 'emission', 'temperature',
            'greenhouse gas', 'ipcc', 'sea level', 'renewable energy', 'fossil fuel',
            'cop[0-9]*', 'paris agreement', 'extreme weather', 'drought', 'flood',
            'methane', 'net zero', 'decarbon', 'climate crisis', 'anthropocene',
            'tipping point', 'climate model', 'climate science', 'climate data',
            'climate impact', 'climate disaster', 'climate risk', 'climate policy',
            'climate legislation', 'climate denial', 'climate skeptic'
        }
        
        # 2. Environmental impact keywords
        impact_keywords = {
            'biodiversity loss', 'ecosystem collapse', 'arctic melt', 'glacial retreat',
            'coral bleaching', 'wildfire', 'heatwave', 'hurricane', 'typhoon', 'cyclone',
            'air quality', 'ocean acidification', 'permafrost', 'species extinction',
            'food security', 'climate refugee', 'climate migration'
        }
        
        # 3. Climate solutions/technology
        solution_keywords = {
            'solar power', 'wind energy', 'ev transition', 'carbon capture',
            'ccs', 'geoengineering', 'climate tech', 'sustainable', 'decarbonization',
            'carbon tax', 'carbon price', 'climate finance', 'green investment',
            'climate adaptation', 'mitigation', 'resilience'
        }
        
        # 4. Common climate hashtags (without #)
        climate_hashtags = {
            'climatechange', 'globalwarming', 'climateaction', 'climatecrisis',
            'climateemergency', 'netzero', 'actonclimate', 'climatejustice',
            'climatehoax', 'climatescam', 'climatecult', 'fridaysforfuture'
        }
        
        # 5. Climate organization references
        org_keywords = {
            'ipcc', 'noaa', 'nasa climate', 'unfccc', 'world meteorological organization',
            'cop', 'unep', 'greenpeace', '350.org', 'sierra club', 'wwf', 'climate reality'
        }
        
        # 6. Negative keywords to exclude financial/scam content
        exclusion_patterns = {
            r'\$\w+',  # Stock tickers
            r'\b(price|stock|invest|profit|dividend|trading|multiplier|rewards)\b',
            r'\b(earn|points|bonus|promo|giveaway|contest|airdrop)\b',
            r'\b(crypto|nft|blockchain|defi|web3|token)\b'
        }
        
        # Check for exclusion patterns first
        for pattern in exclusion_patterns:
            if re.search(pattern, text_lower):
                return False
        
        # Combine all climate-related patterns
        climate_patterns = (
            science_keywords | 
            impact_keywords | 
            solution_keywords | 
            climate_hashtags | 
            org_keywords
        )
        
        # Check for at least 2 distinct climate references
        matches = set()
        for pattern in climate_patterns:
            if re.search(r'\b' + re.escape(pattern) + r'\b', text_lower):
                matches.add(pattern)
                if len(matches) >= 2:  # Require multiple climate references
                    return True
        
        # Additional check for URLs containing climate terms
        url_pattern = r'(https?://\S+)'
        urls = re.findall(url_pattern, text_lower)
        climate_url_indicators = {'climate', 'warming', 'carbon', 'ipcc', 'cop'}
        if any(any(indicator in url for indicator in climate_url_indicators) for url in urls):
            return True
        
        return False
    
    def scrape_climate_tweets(self, max_tweets=100, start_date=None, end_date=None):
        """
        Scrape climate-related tweets with images
        
        Parameters:
        - max_tweets: Maximum number of tweets to collect (including previously collected)
        - start_date: Search start date in format YYYY-MM-DD
        - end_date: Search end date in format YYYY-MM-DD
        
        Returns: DataFrame of scraped climate tweets with images
        """
        # Enhanced climate-related search terms
        climate_terms = [
            # Scientific terms
            "climate change", "global warming", "anthropocene", "CO2 emissions",
            "carbon footprint", "sea level rise", "extreme weather",
            # Policy/Activism terms
            "net zero", "energy transition", "climate policy", "green new deal",
            "climate legislation", "carbon tax", "renewable energy",
            # Misinformation-related terms
            "climate hoax", "climate scam", "climate alarmism", "climate lockdowns",
            "climate cult", "global cooling", "climate overreach",
            # Impact terms
            "climate disaster", "climate migration", "food security", "ecosystem collapse",
            "Arctic melting", "coral bleaching", "wildfire smoke"
        ]
        
        # Enhanced climate-related hashtags
        climate_hashtags = [
            # Mainstream discourse
            "ClimateChange", "GlobalWarming", "ClimateAction", "ClimateCrisis",
            "ClimateEmergency", "NetZero", "IPCC", "COP28",
            # Solutions-focused
            "RenewableEnergy", "SolarPower", "WindEnergy", "EVTransition",
            # Skeptic/Contrarian
            "ClimateScam", "ClimateHoax", "FridaysForHypocrisy", "ClimateRealism",
            "ClimateTruth", "EnergyPoverty", 
            # Activism/Counter-movements
            "ClimateStrike", "ExtinctionRebellion", "JustStopOil", "ClimateDefiance",
            # Geoengineering debate
            "SolarGeoengineering", "CO2Removal", "CarbonCapture"
        ]
        
        # Build search query
        search_url = self.build_search_query(
            search_terms=climate_terms,
            hashtags=climate_hashtags,
            start_date=start_date,
            end_date=end_date
        )
        
        # Scrape tweets
        return self.scrape_tweets(search_url, max_tweets=max_tweets)
    
    def save_data(self, df):
        """Save the scraped data to CSV and download images"""
        if df.empty:
            print("No data to save.")
            return
        
        # Save DataFrame to CSV
        csv_path = os.path.join(self.output_dir, 'climate_tweets.csv')
        df.to_csv(csv_path, index=False)
        print(f"Data saved to {csv_path}")
        
        # Create images directory
        images_dir = os.path.join(self.output_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        # Download images
        print("Downloading images...")
        new_downloads = 0
        
        for i, row in df.iterrows():
            try:
                tweet_hash = row['tweet_hash']
                
                # Skip if this image was already downloaded
                if tweet_hash in self.collected_tweets and self.collected_tweets[tweet_hash].get('image_downloaded', False):
                    continue
                
                img_url = row['image_url']
                if not img_url:
                    continue
                
                # Generate filename from hash
                filename = f"{tweet_hash}.jpg"
                
                # Download image
                img_path = os.path.join(images_dir, filename)
                
                # Skip if file already exists
                if os.path.exists(img_path):
                    self.collected_tweets[tweet_hash]['image_downloaded'] = True
                    continue
                
                import requests
                response = requests.get(img_url, stream=True)
                if response.status_code == 200:
                    with open(img_path, 'wb') as f:
                        f.write(response.content)
                    
                    # Mark as downloaded in tracking data
                    self.collected_tweets[tweet_hash]['image_downloaded'] = True
                    new_downloads += 1
                    
                    if new_downloads % 10 == 0:
                        print(f"Downloaded {new_downloads} new images...")
                else:
                    print(f"Failed to download image {i+1}: HTTP {response.status_code}")
            
            except Exception as e:
                print(f"Error downloading image {i+1}: {str(e)}")
        
        # Save updated tracking data
        self._save_tracking_data()
        
        print(f"Downloaded {new_downloads} new images. All data and images saved to {self.output_dir}")
    
    def close(self):
        """Close the browser and release resources"""
        if self.driver:
            self.driver.quit()
        print("Browser closed.")

def main():
    """Main function to run the Twitter climate scraper"""
    parser = argparse.ArgumentParser(description='Scrape climate-related tweets with images from Twitter')
    
    parser.add_argument('--username', '-u', required=True, help='Twitter username/email')
    parser.add_argument('--password', '-p', required=True, help='Twitter password')
    parser.add_argument('--max_tweets', '-m', type=int, default=100, help='Maximum number of tweets to collect')
    parser.add_argument('--start_date', '-s', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', '-e', help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', default='/Volumes/T7/Climmeme/twitter_data', help='Output directory path')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no browser UI)')
    
    args = parser.parse_args()
    
    # Initialize scraper
    scraper = TwitterClimateScraper(args.username, args.password, output_dir=args.output, headless=args.headless)
    
    try:
        # Login to Twitter
        if not scraper.login():
            print("Login failed. Exiting.")
            return
        
        # Scrape climate tweets
        df = scraper.scrape_climate_tweets(
            max_tweets=args.max_tweets,
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        # Save scraped data
        scraper.save_data(df)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    finally:
        # Clean up
        scraper.close()

if __name__ == "__main__":
    main()