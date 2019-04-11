# Smart Cart

Smart Cart is a comprehensive grocery list manager that offers smart recommendations on things you may want to add to your list!

Check out the [Project Charter](CHARTER.md) for some background on this project's inception.

Or, to see the planned work, check out the [issues](https://github.com/michaelfedell/smart_cart/issues) or [ZenHub Board](https://github.com/michaelfedell/smart_cart#workspaces/smart-cart-5cae419280656854a0156607/board?repos=180641233)

# Charter

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

### Theme 1: Data

- Epic 1: Data Management and Preparation
  - Exploration of available data
  - Unite data in single source for application use
  - Text Cleaning
  - Move from CSV to RDB for speed, persistence, convenience
  - Fuzzy Matching

- Epic 2: Recommender System for List Additions
  - Rule Mining (market basket analysis of all shopper data)
  - Historical (common items in user's history)
  - Collaborative Filtering (suggestions from similar users)
  - Prioritize Recommendations
  - Optimize queries
  - Recommendation Feedback/Tracking

### Theme 2: Users

- Epic 3: User Management
  - Account Schema (prepare datastore in postgres)
  - New User Signup Flow
  - Login/Logout Functionality
  - Profile/Settings

- Epic 4: Social
  - Friends
  - Shared Lists
  - OAuth

### Theme 3: Interface

- Epic 5: Lists
  - Reusable List Item Component
  - Archive for Completed Items
  - List Ordering (drag & drop or sort by category etc)
  - Custom Colors/Styling
  - Click to Add Recommendation
  - Save List for Later
  - Multiple Lists

- Epic 6: Recipes
  - Source Recipe Data
  - Normalize Recipe Ingredients
  - Browse Popular Recipes
  - Send Recipe to List
  - Saved Recipes
  - Likely Recipe Matching
  - Show Recipes Button
  - Recipe Instruction Interface

## Backlog

1. Preliminary EDA (1)
2. Joins/Unions (2)
3. Text Cleaning (5)
4. Move from CSV to RDB (2)
5. Rule Mining (market basket analysis) (13)
6. Collaborative Filtering (8)
7. Historical (common items) (5)
8. Prioritize Recommendations (3)
9. Account Information (database) (3)
10. New User (5)
11. Login/Logout (8)
12. Shared Lists (3)
13. OAuth (3)
14. Profile/Settings (8)
15. Reusable List Item Component (13)
16. Archive for Completed Items (8)
17. List Ordering (drag & drop or sort by category etc) (8)
18. Custom Colors/Styling (3)
19. Click to Add Recommendation (5)
20. Save List for Later (8)
21. Source Recipe Data (3)
22. Browse Popular Recipes (3)
23. Send Recipe to List (5)
24. Saved Recipes (3)

## Icebox

- 1.1.Fuzzy Matching
- 1.2.Recommendation Feedback/Tracking
- 1.2.Optimize Queries
- 2.4.Friends
- 3.5.Multiple Lists
- 3.6.Normalize Recipe Ingredients
- 3.6.Show Recipes Button
- 3.6 Likely Recipe Matching
- 3.6.Recipe Instruction Interface

---

## Structure

## Documentation

## Getting Started

## Testing

## Acknowledgements

## DataLinks

- "The Instacart Online Grocery Shopping Dataset 2017", Accessed from <https://www.instacart.com/datasets/grocery-shopping-2017> on 2019-04-10