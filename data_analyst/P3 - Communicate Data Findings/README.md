# Ford GoBike System Data
## by LongTH


## Dataset
**Ford GoBike System Dataset**  includes information about individual rides made in a bike-sharing system covering the greater San Francisco Bay area.
(For more information, Ford GoBike is a regional public bicycle sharing system in the San Francisco Bay Area, California. Beginning operation in August 2013 as Bay Area Bike Share, the Ford GoBike system currently has over 2,600 bicycles in 262 stations across San Francisco, East Bay and San Jose. On June 28, 2017, the system officially launched as Ford GoBike in a partnership with Ford Motor Company.)

The dataset orginally consists of 183,412 row (records) and 16 columns.
All columns/features contain information about three major points as follow: 
>        
    - trip (duration_sec, start_time, end_time)
    - station (start_station_id, start_station_name, start_station_latitude, start_station_longitude, end_station_id, end_station_name, end_station_latitude, end_station_longitude)
    - User (bike_id, user_type, member_birth_year, member_gender, bike_share_for_all_trip)
 


## Summary of Findings

>
    - Most riders are Subscribers
    - Most riders are Male and male riders tends to be older than female riders
    - Most of trips are conducted from young riders (< 45 years old)
    - Most of trips are taking less than 1000seconds .
    - There are less trips in the weekend than in the weekdays. However, trip in the weekends have longer duration on average than in the weekdays
    - The hours in day that have the most number of trips are 8a.m and 5p.m, while the number of the trips is the least and decreasing rapidly after midnight until dawn
    - Subscribers tends to ride shorter duration trip than Customers. However, old Subscribers have longer duration trip than old Customers 
    - There are some correlation between trip duration and trip distance length

## Key Insights for Presentation

> User type and their trip duration
- Most users are Subscribers
- Old Subcribers have ride longer than old Customers despite of average statistics
    
> User age and their trip duration
- Most users are less then 45 years old
- Younger riders have longer trip than older