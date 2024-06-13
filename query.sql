WITH CommentRank AS (
    SELECT
        tc.ticket_id,
        tc.created AS comment_created,
        FORMAT(
            'Ticket Comment %d\nCreated On : %s\nCreated By : %s (%s), Role: %s\n%s\n%s\n%s\n',
            ROW_NUMBER() OVER(PARTITION BY tc.ticket_id ORDER BY tc.created),
            CAST(tc.created AS VARCHAR),
            u.name,
            u.email,
            u.role,
            '-------------------------',
            tc.body,
            '-------------------------'
        ) AS formatted_comment
    FROM
        TICKET_COMMENT tc
    JOIN
        USER u ON tc.user_id = u.id
)
SELECT
    t.id as ticket_id,
    t.subject as ticket_subject,
    t.created_at as ticket_created_at,
    FORMAT(
        '\nTicket ID : %d\nCreated On : %s\nSubject : %s\nStatus : %s\nPriority : %s\n\n%s',
        t.id,
        CAST(t.created_at AS VARCHAR),
        t.subject,
        t.status,
        t.priority,
        LISTAGG(cr.formatted_comment, '\n') WITHIN GROUP (ORDER BY cr.comment_created)
    ) AS ticket_details
FROM
    TICKET t
LEFT JOIN
    CommentRank cr ON t.id = cr.ticket_id
GROUP BY
    t.id, t.subject, t.created_at, t.status, t.priority
LIMIT 10