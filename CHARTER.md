# Smart Cart Project Charter

This project was started as a way to improve the grocery shopping experience, especially for young people who are still learning to cook. The project also serves as a testing ground for various tools and technologies such as cloud computing, machine-learning driven recommendation systems, application deployment and maintenance, as well as basic web development in the Flask-python ecosystem.

## Vision

To create a simple and fun experience out of grocery shopping which inspires creativity and exploration.

## Mission

Learn from experienced cooks and grocery shoppers and provide the resulting guidance in a sleek application that makes it easy for users to manage grocery lists, find new recipe ideas, and share their findings with friends.

## Success Criteria

- [] 30% utilization of all served recommendations
- [] 75% of users adopt recommendations
- [] ≥ 80% positive feedback on UX design
- [] manage grocery lists and gather feedback from at least 20 total users

## Planned Work

[Click here](https://github.com/michaelfedell/smart_cart/issues) for the live issue board

### Data Management

Backlog:

- Exploration of available data
- Unite data in single source for application use
- Text Cleaning
- Move from CSV to RDB for speed, persistence, convenience

Icebox:

- Fuzzy Matching

### Rec System for List Additions

Backlog:

- Rule Mining (market basket analysis of all shopper data)
- Historical (common items in user's history)
- Collaborative Filtering (suggestions from similar users)
- Prioritize Recommendations

Icebox:

- Optimize queries
- Recommendation Feedback/Tracking

### User Management

Backlog:

- Account Schema (prepare datastore in postgres)
- New User Signup Flow
- Login/Logout Functionality
- Profile/Settings

### Social Features

- Shared Lists
- OAuth

Icebox:

- Friends

### List Interface

Backlog:

- Reusable List Item Component
- Archive for Completed Items
- List Ordering (drag & drop or sort by category etc)
- Custom Colors/Styling
- Click to Add Recommendation
- Save List for Later

Icebox:

- Multiple Lists
- Shared Lists

### Recipes

Backlog:

- Source Recipe Data
- Browse Popular Recipes
- Send Recipe to List
- Saved Recipes

Icebox:

- Normalize Recipe Ingredients
- Show Recipes Button
- Likely Recipe Matching
- Recipe Instruction Interface
