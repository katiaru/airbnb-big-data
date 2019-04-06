listings_columns = ['id', 'latitude', 'longitude', 'amenities',
                    'security_deposit', 'cleaning_fee', 'neighbourhood_cleansed', 'bed_type', 'experiences_offered',
                    'host_verifications', 'review_scores_location', 'cancellation_policy', 'room_type',
                    'reviews_per_month', 'accommodates', 'review_scores_rating', 'host_is_superhost',
                    'host_listings_count', 'availability_30', 'price']#'first_review', 'last_review','number_of_reviews'

categorical_columns = ['cancellation_policy', 'room_type', 'host_is_superhost']

continuous_columns = ['latitude', 'longitude', 'security_deposit', 'cleaning_fee', 'review_scores_location',
                      'reviews_per_month', 'accommodates', 'review_scores_rating','host_listings_count',
                      'availability_30']
