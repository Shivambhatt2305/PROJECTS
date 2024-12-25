# from flask import Flask, render_template, request, redirect, url_for

# app = Flask(__name__)

# # Store reviews and ratings in memory (could use a database for persistence)
# reviews = []
# ratings = [0, 0, 0, 0, 0]  # Stores count for each rating (1 to 5 stars)

# @app.route('/')
# def index():
#     # Calculate average rating and rating percentages
#     total_reviews = len(reviews)
#     total_rating = sum([review['rating'] for review in reviews])
#     average_rating = (total_rating / total_reviews) if total_reviews > 0 else 0

#     # Calculate percentage for each rating level
#     rating_percentages = [(count / total_reviews) * 100 if total_reviews > 0 else 0 for count in ratings]

#     return render_template(
#         'index.html',
#         reviews=reviews,
#         ratings=ratings,
#         average_rating=round(average_rating, 1),
#         rating_percentages=rating_percentages,
#         total_reviews=total_reviews
#     )

# @app.route('/submit_review', methods=['POST'])
# def submit_review():
#     if request.method == 'POST':
#         name = request.form['name']
#         review_text = request.form['review']
#         rating = 3  # Default to 3 stars (Neutral)

#         # Automatic rating based on review text
#         if 'excellent' in review_text.lower() or 'great' in review_text.lower():
#             rating = 5
#         elif 'good' in review_text.lower() or 'satisfactory' in review_text.lower():
#             rating = 4
#         elif 'fair' in review_text.lower() or 'okay' in review_text.lower():
#             rating = 3
#         elif 'bad' in review_text.lower() or 'poor' in review_text.lower():
#             rating = 1

#         # Create a new review and append it to the reviews list
#         new_review = {
#             'name': name,
#             'rating': rating,
#             'review': review_text
#         }

#         reviews.append(new_review)
#         ratings[rating - 1] += 1  # Update the rating count for the specific rating

#         # Redirect to the main page to show the updated review
#         return redirect(url_for('index'))

# @app.route('/send_feedback', methods=['POST'])
# def send_feedback():
#     additional_feedback = request.form.get('additional_feedback')
#     review_id = request.form.get('review_id')

#     if not additional_feedback or not review_id:
#         return redirect(url_for('index'))

#     try:
#         review_id = int(review_id)
#         if review_id < 0 or review_id >= len(reviews):
#             return redirect(url_for('index'))  # Invalid review ID

#         # Get the specific review and add additional feedback
#         review = reviews[review_id]
#         review['additional_feedback'] = additional_feedback

#         # Instead of sending feedback via email, just update the review
#         return redirect(url_for('index'))
    
#     except ValueError:
#         return redirect(url_for('index'))  # Invalid review ID

# if __name__ == "__main__":
#     app.run(debug=True)


#2
from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for flashing messages

# Store reviews and ratings in memory (could use a database for persistence)
reviews = []
ratings = [0, 0, 0, 0, 0]  # Stores count for each rating (1 to 5 stars)

@app.route('/')
def college_ui():
    return render_template('collegeui.html')

@app.route('/reviews')
def index():
    # Calculate average rating and rating percentages
    total_reviews = len(reviews)
    total_rating = sum([review['rating'] for review in reviews])
    average_rating = (total_rating / total_reviews) if total_reviews > 0 else 0

    # Calculate percentage for each rating level
    rating_percentages = [(count / total_reviews) * 100 if total_reviews > 0 else 0 for count in ratings]

    return render_template(
        'index.html',
        reviews=reviews,
        ratings=ratings,
        average_rating=round(average_rating, 1),
        rating_percentages=rating_percentages,
        total_reviews=total_reviews
    )

@app.route('/submit_review', methods=['POST'])
def submit_review():
    if request.method == 'POST':
        name = request.form['name']
        review_text = request.form['review']
        rating = 3  # Default to 3 stars (Neutral)

        # List of patterns that may indicate a fake review
        fake_review_indicators = [
        "asdf", "qwerty", "12345", "zxczxc", "testtest", "lorem ipsum", "111111",
        "aaaaa", "zzzzz", "random text", "blah blah", "testing123", "abc123","eeeee",
        "fake", "scam", "not real", "fraudulent", "shady", "bogus", "hoax",
        "misleading", "paid review", "sponsored", "deceptive", "untrustworthy",
        "repeat", "excellent excellent excellent", "very very good", "amazing amazing",
        "best best best", "perfect perfect perfect", "five stars five stars"
    ]


        # Check for fake review indicators
        for indicator in fake_review_indicators:
            if indicator in review_text.lower():
                flash("Please provide an honest review. Fake reviews are not allowed.", "error")
                return redirect(url_for('college_ui'))

        # Automatic rating based on review text
        if 'excellent' in review_text.lower() or 'great' in review_text.lower():
            rating = 5
        elif 'good' in review_text.lower() or 'satisfactory' in review_text.lower():
            rating = 4
        elif 'fair' in review_text.lower() or 'okay' in review_text.lower():
            rating = 3
        elif 'bad' in review_text.lower() or 'poor' in review_text.lower():
            rating = 1

        # Create a new review and append it to the reviews list
        new_review = {
            'name': name,
            'rating': rating,
            'review': review_text
        }

        reviews.append(new_review)
        ratings[rating - 1] += 1  # Update the rating count for the specific rating

        # Redirect to the main page to show the updated review
        return redirect(url_for('index'))

@app.route('/send_feedback', methods=['POST'])
def send_feedback():
    additional_feedback = request.form.get('additional_feedback')
    review_id = request.form.get('review_id')

    if not additional_feedback or not review_id:
        return redirect(url_for('index'))

    try:
        review_id = int(review_id)
        if review_id < 0 or review_id >= len(reviews):
            return redirect(url_for('index'))  # Invalid review ID

        # Get the specific review and add additional feedback
        review = reviews[review_id]
        review['additional_feedback'] = additional_feedback

        # Instead of sending feedback via email, just update the review
        return redirect(url_for('index'))

    except ValueError:
        return redirect(url_for('index'))  # Invalid review ID

if __name__ == "__main__":
    app.run(debug=True)


#3
# from flask import Flask, render_template, request, redirect, url_for

# app = Flask(__name__)

# # Store reviews and ratings in memory (could use a database for persistence)
# reviews = []
# ratings = [0, 0, 0, 0, 0]  # Stores count for each rating (1 to 5 stars)

# @app.route('/')
# def college_ui():
#     return render_template('collegeui.html')

# @app.route('/reviews')
# def index():
#     # Calculate average rating and rating percentages
#     total_reviews = len(reviews)
#     total_rating = sum([review['rating'] for review in reviews])
#     average_rating = (total_rating / total_reviews) if total_reviews > 0 else 0

#     # Calculate percentage for each rating level
#     rating_percentages = [(count / total_reviews) * 100 if total_reviews > 0 else 0 for count in ratings]

#     return render_template(
#         'index.html',
#         reviews=reviews,
#         ratings=ratings,
#         average_rating=round(average_rating, 1),
#         rating_percentages=rating_percentages,
#         total_reviews=total_reviews
#     )

# @app.route('/submit_review', methods=['POST'])
# def submit_review():
#     if request.method == 'POST':
#         name = request.form['name']
#         review_text = request.form['review']
#         rating = 3  # Default to 3 stars (Neutral)

#         # List of patterns that may indicate a fake review
#         fake_review_indicators = [
#             "asdf", "qwerty", "12345",  # Gibberish patterns
#             "fake", "scam", "not real",  # Obvious fake indicators
#             "repeat", "excellent excellent excellent",  # Repetition
#         ]

#         # Check for fake review indicators
#         for indicator in fake_review_indicators:
#             if indicator in review_text.lower():
#                 warning_message = "Please provide an honest review. Fake reviews are not allowed."
#                 return render_template('collegeui.html', warning_message=warning_message)

#         # Automatic rating based on review text
#         if 'excellent' in review_text.lower() or 'great' in review_text.lower():
#             rating = 5
#         elif 'good' in review_text.lower() or 'satisfactory' in review_text.lower():
#             rating = 4
#         elif 'fair' in review_text.lower() or 'okay' in review_text.lower():
#             rating = 3
#         elif 'bad' in review_text.lower() or 'poor' in review_text.lower():
#             rating = 1

#         # Create a new review and append it to the reviews list
#         new_review = {
#             'name': name,
#             'rating': rating,
#             'review': review_text
#         }

#         reviews.append(new_review)
#         ratings[rating - 1] += 1  # Update the rating count for the specific rating

#         # Redirect to the main page to show the updated review
#         return redirect(url_for('index'))

# @app.route('/send_feedback', methods=['POST'])
# def send_feedback():
#     additional_feedback = request.form.get('additional_feedback')
#     review_id = request.form.get('review_id')

#     if not additional_feedback or not review_id:
#         return redirect(url_for('index'))

#     try:
#         review_id = int(review_id)
#         if review_id < 0 or review_id >= len(reviews):
#             return redirect(url_for('index'))  # Invalid review ID

#         # Get the specific review and add additional feedback
#         review = reviews[review_id]
#         review['additional_feedback'] = additional_feedback

#         # Instead of sending feedback via email, just update the review
#         return redirect(url_for('index'))
    
#     except ValueError:
#         return redirect(url_for('index'))  # Invalid review ID

# if __name__ == "__main__":
#     app.run(debug=True)
